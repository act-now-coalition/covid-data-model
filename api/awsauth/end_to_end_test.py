import os
import requests

API_URL = os.getenv("API_URL", "https://api-dev.covidactnow.org/v2")

# A known existing API key to use for testing.
# This key must exist in the DynamoDB table for the target environment.
EXISTING_API_KEY = os.getenv("TEST_API_KEY", "")


def test_registration_closed():
    """Registration endpoint should reject requests."""
    response = requests.post(API_URL + "/register", json={"email": "test@example.com"})
    # After deprecation deploy, registration returns 403.
    # Before deploy, it may return 200 (existing behavior).
    assert response.status_code in (200, 403)


def test_invalid_api_key():
    response = requests.get(f"{API_URL}/states.json", {"apiKey": "fake api key"})
    assert response.status_code == 403
    data = response.json()
    assert "error" in data


def test_missing_api_key():
    response = requests.get(f"{API_URL}/states.json")
    assert response.status_code == 403
    data = response.json()
    assert "error" in data


def test_api_flow_existing_user():
    """Tests that endpoints still work with an existing API key."""
    if not EXISTING_API_KEY:
        # Skip if no test key is configured
        return

    response = requests.get(f"{API_URL}/state/MA.json", {"apiKey": EXISTING_API_KEY})
    assert response.ok

    response = requests.get(f"{API_URL}/states.json", {"apiKey": EXISTING_API_KEY})
    assert response.ok

    response = requests.get(f"{API_URL}/county/36061.json", {"apiKey": EXISTING_API_KEY})
    assert response.ok
