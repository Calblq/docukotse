import jwt
import pytest

from app.security import create_access_token, decode_access_token, hash_password, verify_password


def test_password_hash_roundtrip():
    hashed = hash_password("s3cret")
    assert hashed != "s3cret"
    assert verify_password("s3cret", hashed)
    assert not verify_password("wrong", hashed)


def test_token_roundtrip():
    claims = decode_access_token(create_access_token(7, "staff"))
    assert claims["sub"] == "7"
    assert claims["role"] == "staff"


def test_tampered_token_rejected():
    token = create_access_token(7, "staff")
    with pytest.raises(jwt.PyJWTError):
        decode_access_token(token[:-2] + ("AA" if not token.endswith("AA") else "BB"))
