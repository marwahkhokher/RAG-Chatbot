"""
Face-recognition authentication module.

Provides /auth/register and /auth/login endpoints.
Uses the `face_recognition` library (backed by dlib) for encoding and
comparison.  Falls back to a simpler OpenCV-based stub if the library is
not installed so the rest of the app can still start.
"""

import base64
from io import BytesIO

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app import config
from app import qdrant_rest as qdrant

# ---------------------------------------------------------------------------
# Try to import face_recognition; set a flag so callers know what's available
# ---------------------------------------------------------------------------
# NOTE: a broad guard is deliberate. On Python 3.12+ `face_recognition` can fail
# in ways other than ImportError — e.g. `face_recognition_models` imports
# `pkg_resources` (missing when setuptools isn't installed), and on that failure
# face_recognition calls `quit()`, which raises SystemExit (a BaseException, not
# an Exception). Catching only ImportError let that escape and crashed pytest
# collection. Catch anything so the app degrades gracefully instead.
try:
    import face_recognition  # type: ignore

    FACE_LIB_AVAILABLE = True
except BaseException:  # noqa: BLE001 - intentional: keep the app importable no matter how the lib fails
    FACE_LIB_AVAILABLE = False

# ---------------------------------------------------------------------------
# Persistence – Qdrant Vector Database
# ---------------------------------------------------------------------------

def _ensure_faces_db() -> None:
    # 128 is the standard dimension for face_recognition encodings
    qdrant.ensure_collection(
        vector_size=128, 
        collection_name=config.QDRANT_FACES_COLLECTION, 
        distance="Euclid"
    )

def _get_all_users() -> list[str]:
    _ensure_faces_db()
    with qdrant._client() as client:
        # Simple scroll to get all distinct usernames
        res = client.post(
            f"/collections/{config.QDRANT_FACES_COLLECTION}/points/scroll",
            json={"limit": 100, "with_payload": True, "with_vector": False}
        )
        if res.status_code == 200:
            points = res.json()["result"]["points"]
            usernames = {p["payload"]["username"] for p in points if "username" in p["payload"]}
            return list(usernames)
    return []


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _bytes_to_image(raw: bytes) -> np.ndarray:
    """Decode raw image bytes (PNG/JPEG) to an RGB numpy array.

    Used by the Streamlit UI, whose camera widget hands back raw bytes.
    """
    # Use PIL to decode – it's already a dependency of face_recognition /
    # sentence-transformers so guaranteed to be present.
    from PIL import Image

    img = Image.open(BytesIO(raw)).convert("RGB")
    return np.array(img)


def _b64_to_image(data_url: str) -> np.ndarray:
    """Convert a data-URL or raw base64 string to an RGB numpy array."""
    # Strip the optional "data:image/…;base64," prefix
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    return _bytes_to_image(base64.b64decode(data_url))


def _get_encoding(img_array: np.ndarray) -> list[float] | None:
    """Return 128-d face encoding or None if no face found."""
    if not FACE_LIB_AVAILABLE:
        raise RuntimeError(
            "face_recognition library is not installed. "
            "Install it with: pip install face_recognition"
        )
    encodings = face_recognition.face_encodings(img_array)
    if not encodings:
        return None
    return encodings[0].tolist()


# Login match threshold (Euclidean distance on 128-d encodings; lower = stricter).
FACE_MATCH_THRESHOLD = 0.6


# ---------------------------------------------------------------------------
# Transport-agnostic auth core — shared by the FastAPI routes below AND the
# Streamlit UI (streamlit_app.py). Both take an already-decoded RGB array and
# return an AuthResponse-shaped dict. They raise RuntimeError only when the
# face library is unavailable, leaving the caller to decide how to surface it.
# ---------------------------------------------------------------------------

def register_face(username: str, img_array: np.ndarray) -> dict:
    """Encode the face in `img_array` and store it under `username`."""
    encoding = _get_encoding(img_array)
    if encoding is None:
        return {
            "authenticated": False,
            "username": "",
            "message": "No face detected. Please ensure your face is clearly visible.",
        }

    username = username.strip()
    _ensure_faces_db()
    qdrant.upsert_points(
        vectors=[encoding],
        payloads=[{"username": username}],
        collection_name=config.QDRANT_FACES_COLLECTION
    )

    return {
        "authenticated": True,
        "username": username,
        "message": f"Face registered successfully for {username}!",
    }


def login_face(img_array: np.ndarray) -> dict:
    """Match the face in `img_array` against the registered DB."""
    encoding = _get_encoding(img_array)
    if encoding is None:
        return {
            "authenticated": False,
            "username": "",
            "message": "No face detected. Please look at the camera.",
        }

    _ensure_faces_db()
    
    # Search for the closest match in Qdrant
    try:
        results = qdrant.search(
            vector=encoding, 
            k=1, 
            collection_name=config.QDRANT_FACES_COLLECTION
        )
    except Exception: # noqa: BLE001
        results = []

    if not results:
        return {
            "authenticated": False,
            "username": "",
            "message": "No registered users yet. Please register first.",
        }

    best_match_point = results[0]
    best_distance = best_match_point["score"]
    best_match = best_match_point["payload"].get("username")

    if best_match and best_distance < FACE_MATCH_THRESHOLD:
        return {
            "authenticated": True,
            "username": best_match,
            "message": f"Welcome back, {best_match}!",
        }

    return {
        "authenticated": False,
        "username": "",
        "message": "Face not recognized. Please register or try again.",
    }


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
router = APIRouter(prefix="/auth", tags=["auth"])


class RegisterRequest(BaseModel):
    username: str
    image: str  # base64 / data-URL of webcam frame


class LoginRequest(BaseModel):
    image: str  # base64 / data-URL of webcam frame


class AuthResponse(BaseModel):
    authenticated: bool
    username: str = ""
    message: str = ""


@router.post("/register", response_model=AuthResponse)
def register(req: RegisterRequest):
    if not req.username.strip():
        raise HTTPException(status_code=400, detail="Username cannot be empty")

    try:
        img = _b64_to_image(req.image)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}")

    try:
        result = register_face(req.username, img)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return AuthResponse(**result)


@router.post("/login", response_model=AuthResponse)
def login(req: LoginRequest):
    try:
        img = _b64_to_image(req.image)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}")

    try:
        result = login_face(img)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return AuthResponse(**result)


@router.get("/status")
def auth_status():
    """Check whether the face recognition backend is available."""
    return {
        "face_recognition_available": FACE_LIB_AVAILABLE,
        "registered_users": _get_all_users(),
    }
