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


def ensure_collection(vector_size: int, collection_name: str = config.QDRANT_COLLECTION, distance: str = "Cosine") -> None:
    with _client() as client:
        res = client.get(f"/collections/{collection_name}")
        if res.status_code == 200:
            return
        create = client.put(
            f"/collections/{collection_name}",
            json={"vectors": {"size": vector_size, "distance": distance}},
        )
        create.raise_for_status()


def upsert_points(vectors: list[list[float]], payloads: list[dict], collection_name: str = config.QDRANT_COLLECTION) -> None:
    points = [
        {"id": str(uuid.uuid4()), "vector": vector, "payload": payload}
        for vector, payload in zip(vectors, payloads)
    ]
    with _client() as client:
        res = client.put(
            f"/collections/{collection_name}/points",
            params={"wait": "true"},
            json={"points": points},
        )
        res.raise_for_status()


def search(vector: list[float], k: int = 4, collection_name: str = config.QDRANT_COLLECTION) -> list[dict]:
    with _client() as client:
        res = client.post(
            f"/collections/{collection_name}/points/search",
            json={"vector": vector, "limit": k, "with_payload": True},
        )
        res.raise_for_status()
        return res.json()["result"]
