"""
Talks to Qdrant Cloud over plain HTTPS REST, deliberately avoiding the
qdrant-client package (which pulls in grpc and can be blocked by strict
application-control / antivirus policies on locked-down machines).
"""
import uuid

import httpx

from app import config


def _client() -> httpx.Client:
    return httpx.Client(
        base_url=config.QDRANT_URL,
        headers={"api-key": config.QDRANT_API_KEY},
        timeout=30.0,
    )


def ensure_collection(vector_size: int) -> None:
    with _client() as client:
        res = client.get(f"/collections/{config.QDRANT_COLLECTION}")
        if res.status_code == 200:
            return
        create = client.put(
            f"/collections/{config.QDRANT_COLLECTION}",
            json={"vectors": {"size": vector_size, "distance": "Cosine"}},
        )
        create.raise_for_status()


def upsert_points(vectors: list[list[float]], payloads: list[dict]) -> None:
    points = [
        {"id": str(uuid.uuid4()), "vector": vector, "payload": payload}
        for vector, payload in zip(vectors, payloads)
    ]
    with _client() as client:
        res = client.put(
            f"/collections/{config.QDRANT_COLLECTION}/points",
            params={"wait": "true"},
            json={"points": points},
        )
        res.raise_for_status()


def search(vector: list[float], k: int = 4) -> list[dict]:
    with _client() as client:
        res = client.post(
            f"/collections/{config.QDRANT_COLLECTION}/points/search",
            json={"vector": vector, "limit": k, "with_payload": True},
        )
        res.raise_for_status()
        return res.json()["result"]
