import pytest
import uuid
from backend.app.database.connection import SessionLocal, init_db
from backend.app.database.models import User, Visitor
from backend.app.api.auth_routes import hash_password, verify_password
from backend.app.api.auth_middleware import generate_jwt_token

@pytest.fixture(autouse=True)
def setup_db():
    init_db()

def test_password_hashing():
    pwd = "ForestRangerSecurePass2026!"
    hashed = hash_password(pwd)
    assert verify_password(pwd, hashed) is True
    assert verify_password("WrongPassword", hashed) is False

def test_jwt_token_generation():
    user = User(
        id=str(uuid.uuid4()),
        email="test_ranger@forest.gov",
        password_hash=hash_password("test"),
        role="OFFICER",
        status="ACTIVE"
    )
    token = generate_jwt_token(user)
    assert token is not None
    assert isinstance(token, str)
    assert len(token) > 20
