import os
import requests
import uuid

API_URL = os.getenv("API_URL", "https://api-dev.covidactnow.org/v2")


def test_api_flow():
    randomness = uuid.uuid4().hex[:8]
    test_email = f"testEmail+{randomness}@covidactnow.org"
    response = requests.post(API_URL + "/register", json={"email": test_email})
    assert response.ok

    data = response.json()
    api_key = data["api_key"]
    assert data["new_user"]

    response = requests.get(f"{API_URL}/state/MA.json", {"apiKey": api_key})
    assert response.ok

    response = requests.get(f"{API_URL}/states.json", {"apiKey": api_key})
    assert response.ok

    response = requests.get(f"{API_URL}/county/36061.json", {"apiKey": api_key})
    assert response.ok


def test_blocked_email():
    # Should be set in .env file for local environment
    test_email = "blocked@covidactnow.org"
    response = requests.post(API_URL + "/register", json={"email": test_email})
    assert response.ok
    data = response.json()
    api_key = data["api_key"]

    response = requests.get(f"{API_URL}/state/MA.json", {"apiKey": api_key})
    assert response.status_code == 403
    assert "error" in response.json()


def test_invalid_api_key():
    response = requests.get(f"{API_URL}/states.json", {"apiKey": "fake api key"})
    assert response.status_code == 403
    response = response.json()
    expected_error = {"error": "Invalid API key."}
    assert response == expected_error


def test_api_flow_existing_user():
    # Tests to make sure that endpoints work with an account that should already be in the
    # database. Can be used on on dev or prod url
    email = "chris+testing@covidactnow.org"
    response = requests.post(API_URL + "/register", json={"email": email})
    assert response.ok

    data = response.json()
    api_key = data["api_key"]

    # User should not be new, should be pulled from existing user.
    assert not data["new_user"]

    response = requests.get(f"{API_URL}/state/MA.json", {"apiKey": api_key})
    assert response.ok

    response = requests.get(f"{API_URL}/states.json", {"apiKey": api_key})
    assert response.ok

    response = requests.get(f"{API_URL}/county/36061.json", {"apiKey": api_key})
    assert response.ok
