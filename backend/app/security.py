"""Password hashing and JWT access tokens."""

from datetime import UTC, datetime, timedelta

import bcrypt
import jwt

from app.config import get_settings

_ALGORITHM = "HS256"


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(password.encode(), password_hash.encode())


def create_access_token(user_id: int, role: str) -> str:
    settings = get_settings()
    expires = datetime.now(UTC) + timedelta(minutes=settings.jwt_expire_minutes)
    payload = {"sub": str(user_id), "role": role, "exp": expires}
    return jwt.encode(payload, settings.jwt_secret, algorithm=_ALGORITHM)


def decode_access_token(token: str) -> dict:
    """Return the token claims. Raises jwt.PyJWTError if invalid or expired."""
    return jwt.decode(token, get_settings().jwt_secret, algorithms=[_ALGORITHM])
