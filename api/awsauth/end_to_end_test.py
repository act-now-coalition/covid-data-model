import requests
import uuid

DEV_API_URL = "https://api-dev.covidactnow.org/v2"


def test_api_flow():
    randomness = uuid.uuid4().hex[:8]
    test_email = f"testEmail+{randomness}@covidactnow.org"
    response = requests.post(DEV_API_URL + "/register", json={"email": test_email})
    assert response.ok

    data = response.json()
    api_key = data["api_key"]

    response = requests.get(f"{DEV_API_URL}/state/MA.json", {"apiKey": api_key})
    assert response.ok

    response = requests.get(f"{DEV_API_URL}/states.json", {"apiKey": api_key})
    assert response.ok

    response = requests.get(f"{DEV_API_URL}/county/36061.json", {"apiKey": api_key})
    assert response.ok


def test_blocked_email():
    # Should be set in .env file for local environment
    test_email = "blocked@covidactnow.org"
    response = requests.post(DEV_API_URL + "/register", json={"email": test_email})
    assert response.ok
    data = response.json()
    api_key = data["api_key"]

    response = requests.get(f"{DEV_API_URL}/state/MA.json", {"apiKey": api_key})
    assert response.status_code == 403
    assert "error" in response.json()


def test_invalid_api_key():
    response = requests.get(f"{DEV_API_URL}/states.json", {"apiKey": "fake api key"})
    assert response.status_code == 403
