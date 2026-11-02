VIOS = {
    "plate": "nbc-1234",
    "vin": "MR0FB8CD3K0123456",
    "make": "Toyota",
    "model": "Vios",
    "year": 2019,
}


def test_create_and_get_vehicle(client, staff_headers):
    resp = client.post("/vehicles", json=VIOS, headers=staff_headers)
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["plate"] == "NBC 1234"  # canonicalized
    assert body["status"] == "acquired"

    resp = client.get(f"/vehicles/{body['id']}", headers=staff_headers)
    assert resp.status_code == 200
    assert resp.json()["model"] == "Vios"


def test_list_vehicles(client, staff_headers):
    client.post("/vehicles", json=VIOS, headers=staff_headers)
    client.post("/vehicles", json={"make": "Honda", "model": "City"}, headers=staff_headers)
    resp = client.get("/vehicles", headers=staff_headers)
    assert [v["make"] for v in resp.json()] == ["Toyota", "Honda"]


def test_duplicate_plate_conflicts_across_formats(client, staff_headers):
    assert client.post("/vehicles", json=VIOS, headers=staff_headers).status_code == 201
    dup = {**VIOS, "plate": "NBC 1234"}
    assert client.post("/vehicles", json=dup, headers=staff_headers).status_code == 409


def test_patch_vehicle(client, staff_headers):
    vid = client.post("/vehicles", json=VIOS, headers=staff_headers).json()["id"]
    resp = client.patch(f"/vehicles/{vid}", json={"status": "listed"}, headers=staff_headers)
    assert resp.status_code == 200
    assert resp.json()["status"] == "listed"
    assert resp.json()["plate"] == "NBC 1234"  # untouched fields stay


def test_invalid_status_rejected(client, staff_headers):
    vid = client.post("/vehicles", json=VIOS, headers=staff_headers).json()["id"]
    resp = client.patch(f"/vehicles/{vid}", json={"status": "stolen"}, headers=staff_headers)
    assert resp.status_code == 422


def test_missing_vehicle_404(client, staff_headers):
    assert client.get("/vehicles/999", headers=staff_headers).status_code == 404


def test_only_admin_can_delete(client, staff_headers, admin_headers):
    vid = client.post("/vehicles", json=VIOS, headers=staff_headers).json()["id"]
    assert client.delete(f"/vehicles/{vid}", headers=staff_headers).status_code == 403
    assert client.delete(f"/vehicles/{vid}", headers=admin_headers).status_code == 204
    assert client.get(f"/vehicles/{vid}", headers=admin_headers).status_code == 404
