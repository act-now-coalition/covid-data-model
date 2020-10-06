import requests
import uuid
import json

DEV_API_URL = "https://api-dev.covidactnow.org/v2"


def test_api_flow():
    randomness = uuid.uuid4().hex[:8]
    test_email = f"testEmail+{randomness}@gmail.com"
    response = requests.post(DEV_API_URL + "/register", json={"email": test_email})
    data = response.json()
    assert data["statusCode"] == 200
    api_key = json.loads(data["body"])["api_key"]

    response = requests.get(f"{DEV_API_URL}/states.json", {"apiKey": api_key})
    assert response.ok

    response = requests.get(f"{DEV_API_URL}/state/MA.json", {"apiKey": api_key})
    assert response.ok

    response = requests.get(f"{DEV_API_URL}/county/36061.json", {"apiKey": api_key})
    assert response.ok
