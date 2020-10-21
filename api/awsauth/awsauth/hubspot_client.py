import requests
from awsauth.config import Config


class HubSpotAPICallFailed(Exception):
    """An issue with a HubSpot API call occurred."""


class HubSpotClient:
    def create_contact(self, email: str) -> dict:
        """Creates a contact in hubspot for a given email.

        Args:
            email: Email address of contact.

        Returns: Successful contact creation response.

        Raises: HubSpotAPICallFailed on Hubspot error.
        """
        if not Config.Constants.HUBSPOT_ENABLED:
            return

        # https://pypi.org/project/hubspot-api-client/
        url = "https://api.hubapi.com/crm/v3/objects/contacts"

        querystring = {"hapikey": Config.Constants.HUBSPOT_API_KEY}
        data = {"properties": {"email": email}}

        response = requests.post(url, json=data, params=querystring)

        if not response.ok:
            raise HubSpotAPICallFailed("Hubspot contact creation failed")

        return response.json()
