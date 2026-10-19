import math

import pytest
from fastapi.testclient import TestClient

from app.main import app, validate_values

SAMPLE = [14.23, 1.71, 2.43, 15.6, 127, 2.8, 3.06, 0.28, 2.29, 5.64, 1.04, 3.92, 1065]


@pytest.fixture
def client():
    with TestClient(app) as instance:
        yield instance


@pytest.mark.parametrize("values", [SAMPLE, [0] * 13, [1] * 13])
def test_prediction_contract(client, values):
    response = client.post("/predict", json={"values": values})
    assert response.status_code == 200
    body = response.json()
    assert body["cultivar"] in [1, 2, 3]
    # assert body["cultivar"] in [0, 1, 2] # test
    assert len(body["probabilities"]) == 3
    assert all(0 <= p <= 1 for p in body["probabilities"])
    assert sum(body["probabilities"]) == pytest.approx(1)
    assert body == client.post("/predict", json={"values": values}).json()


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"values": []},
        {"values": [1] * 12},
        {"values": [1] * 14},
        {"values": [-1] * 13},
        {"values": [True] * 13},
        {"values": ["1"] * 13},
        {"values": [None] * 13},
        {"values": [1e20] * 13},
        {"values": SAMPLE, "unexpected": 1},
    ],
)
def test_invalid_payloads(client, payload):
    assert client.post("/predict", json=payload).status_code == 422


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, -0.1, True])
def test_nonfinite_and_negative_guard(bad):
    with pytest.raises(ValueError):
        validate_values([bad] * 13, list(range(13)))


@pytest.mark.parametrize("endpoint", ["/", "/health", "/model", "/docs"])
def test_service_endpoints(client, endpoint):
    assert client.get(endpoint).status_code == 200
    if endpoint == "/health":
        assert client.get(endpoint).json()["run_id"] == client.get("/model").json()["run_id"]
