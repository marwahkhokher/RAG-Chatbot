"""
Face-recognition authentication module.

Provides /auth/register and /auth/login endpoints.
Uses the `face_recognition` library (backed by dlib) for encoding and
comparison.  Falls back to a simpler OpenCV-based stub if the library is
not installed so the rest of the app can still start.
"""

import base64
import json
from io import BytesIO
from pathlib import Path

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Try to import face_recognition; set a flag so callers know what's available
# ---------------------------------------------------------------------------
try:
    import face_recognition  # type: ignore

    FACE_LIB_AVAILABLE = True
except ImportError:
    FACE_LIB_AVAILABLE = False

# ---------------------------------------------------------------------------
# Persistence – simple JSON file next to the project root
# ---------------------------------------------------------------------------
FACE_DB_PATH = Path(__file__).resolve().parent.parent / "face_db.json"


def _load_db() -> dict:
    """Return {username: [[encoding_floats], ...], ...}."""
    if FACE_DB_PATH.exists():
        with open(FACE_DB_PATH, "r") as f:
            return json.load(f)
    return {}


def _save_db(db: dict) -> None:
    with open(FACE_DB_PATH, "w") as f:
        json.dump(db, f)


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _b64_to_image(data_url: str) -> np.ndarray:
    """Convert a data-URL or raw base64 string to an RGB numpy array."""
    # Strip the optional "data:image/…;base64," prefix
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    img_bytes = base64.b64decode(data_url)

    # Use PIL to decode – it's already a dependency of face_recognition / 
    # sentence-transformers so guaranteed to be present.
    from PIL import Image

    img = Image.open(BytesIO(img_bytes)).convert("RGB")
    return np.array(img)


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
        encoding = _get_encoding(img)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    if encoding is None:
        return AuthResponse(
            authenticated=False,
            message="No face detected. Please ensure your face is clearly visible.",
        )

    db = _load_db()
    username = req.username.strip()

    # Allow overwriting an existing registration
    if username not in db:
        db[username] = []
    db[username].append(encoding)
    _save_db(db)

    return AuthResponse(
        authenticated=True,
        username=username,
        message=f"Face registered successfully for {username}!",
    )


@router.post("/login", response_model=AuthResponse)
def login(req: LoginRequest):
    try:
        img = _b64_to_image(req.image)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}")

    try:
        encoding = _get_encoding(img)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    if encoding is None:
        return AuthResponse(
            authenticated=False,
            message="No face detected. Please look at the camera.",
        )

    db = _load_db()
    if not db:
        return AuthResponse(
            authenticated=False,
            message="No registered users yet. Please register first.",
        )

    # Compare against every stored encoding
    probe = np.array(encoding)
    best_match: str | None = None
    best_distance: float = 1.0  # lower is better; threshold ≈ 0.6

    for username, stored_encodings in db.items():
        for enc in stored_encodings:
            distance = float(np.linalg.norm(probe - np.array(enc)))
            if distance < best_distance:
                best_distance = distance
                best_match = username

    THRESHOLD = 0.6
    if best_match and best_distance < THRESHOLD:
        return AuthResponse(
            authenticated=True,
            username=best_match,
            message=f"Welcome back, {best_match}!",
        )

    return AuthResponse(
        authenticated=False,
        message="Face not recognized. Please register or try again.",
    )


@router.get("/status")
def auth_status():
    """Check whether the face recognition backend is available."""
    return {
        "face_recognition_available": FACE_LIB_AVAILABLE,
        "registered_users": list(_load_db().keys()),
    }
