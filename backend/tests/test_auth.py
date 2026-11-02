def test_login_and_me(client, staff_headers):
    resp = client.get("/me", headers=staff_headers)
    assert resp.status_code == 200
    assert resp.json()["email"] == "staff@test.local"
    assert resp.json()["role"] == "staff"


def test_login_wrong_password(client, staff_headers):
    resp = client.post("/auth/login", data={"username": "staff@test.local", "password": "nope"})
    assert resp.status_code == 401


def test_login_unknown_user(client, db):
    resp = client.post("/auth/login", data={"username": "ghost@test.local", "password": "x"})
    assert resp.status_code == 401


def test_protected_routes_need_token(client, db):
    assert client.get("/me").status_code == 401
    assert client.get("/vehicles").status_code == 401
    assert client.get("/documents").status_code == 401


def test_invalid_token_rejected(client, db):
    resp = client.get("/me", headers={"Authorization": "Bearer not-a-token"})
    assert resp.status_code == 401


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "database": True}
