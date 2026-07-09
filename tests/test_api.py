"""
Endpoint contract tests. These use the fake ModelManager + patched
evaluator from conftest.py, so they run without hitting the Hub for
checkpoints or the real reference set -- fast, and safe in CI without
GPU or a populated model repo yet.
"""


def test_root(client):
    resp = client.get("/")
    assert resp.status_code == 200


def test_health(client):
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert isinstance(body["resident_models"], list)


def test_generate_happy_path(client):
    resp = client.post(
        "/api/v1/generate",
        json={"topic": "Sports", "size": 100, "n_samples": 2},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["topic"] == "Sports"
    assert body["size"] == 100
    assert len(body["texts"]) == 2
    assert all(isinstance(t, str) for t in body["texts"])


def test_generate_rejects_invalid_topic(client):
    resp = client.post(
        "/api/v1/generate",
        json={"topic": "NotATopic", "size": 100, "n_samples": 1},
    )
    assert resp.status_code == 422  # Pydantic Literal validation


def test_generate_rejects_invalid_size(client):
    resp = client.post(
        "/api/v1/generate",
        json={"topic": "Sports", "size": 999, "n_samples": 1},
    )
    assert resp.status_code == 422


def test_generate_rejects_too_many_samples(client):
    resp = client.post(
        "/api/v1/generate",
        json={"topic": "Sports", "size": 100, "n_samples": 999},
    )
    assert resp.status_code == 422


def test_evaluate_happy_path(client):
    resp = client.post(
        "/api/v1/evaluate",
        json={"topic": "World", "texts": ["some generated headline"]},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["topic"] == "World"
    assert 0.0 <= body["style_sim"] <= 1.0  # patched to return 0.5


def test_evaluate_rejects_empty_texts(client):
    resp = client.post(
        "/api/v1/evaluate",
        json={"topic": "World", "texts": []},
    )
    assert resp.status_code == 422


def test_drift_happy_path(client):
    resp = client.post(
        "/api/v1/drift",
        json={"topic": "World", "size": 100, "n_hops": 3},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["scores"]) == 3


def test_drift_rejects_too_few_hops(client):
    resp = client.post(
        "/api/v1/drift",
        json={"topic": "World", "size": 100, "n_hops": 1},
    )
    assert resp.status_code == 422
