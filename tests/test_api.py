# tests/test_api.py
"""
Educational Goal:
Demonstrate how to write integration tests for a FastAPI service using TestClient.
In Phase 3 (CI/CD), GitHub Actions will run these tests to ensure our API is 
healthy before allowing a deployment to Render.
"""

from fastapi.testclient import TestClient
from src.api import app


def test_root_endpoint():
    """Verify the root endpoint returns instructions."""
    # We MUST use the `with` context manager so the API's `lifespan` startup event runs.
    with TestClient(app) as client:
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "message" in data


def test_health_endpoint():
    """
    Verify the health check responds. 
    Note: We don't assert model_loaded=True because in a CI pipeline, 
    the model might not be generated yet when tests run. We just care that it doesn't crash.
    """
    with TestClient(app) as client:
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "model_version" in data


def test_predict_endpoint_validation_error():
    """
    Verify our Pydantic check list is working.
    We send a patient record that is missing almost all required feature columns.
    It should immediately reject it with a 422 error, proving we don't need manual validation.
    """
    bad_payload = {
        "records": [
            {"ID": "test_patient_001"}  # Missing rx_ds, A, B, C, etc.
        ]
    }

    with TestClient(app) as client:
        response = client.post("/predict", json=bad_payload)

        # We expect a 422 Unprocessable Entity because the schema contract was violated
        assert response.status_code == 422
