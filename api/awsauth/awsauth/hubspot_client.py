from typing import Optional
import requests
from awsauth.config import Config


class HubSpotAPICallFailed(Exception):
    """An issue with a HubSpot API call occurred."""


class HubSpotClient:
    def submit_reg_form(self, email: str) -> Optional[dict]:
        """Submit hubspot registration form.

        Args:
            email: Email of user creating account.

        Returns: Successful form submission response if hubspot enabled. None if hubspot disable.

        Raises: HubSpotAPICallFailed on Hubspot error.
        """
        if not Config.Constants.HUBSPOT_ENABLED:
            return

        portal_id = Config.Constants.HUBSPOT_PORTAL_ID
        form_guid = Config.Constants.HUBSPOT_REG_FORM_GUID

        # https://legacydocs.hubspot.com/docs/methods/forms/submit_form
        url = f"https://api.hsforms.com/submissions/v3/integration/submit/{portal_id}/{form_guid}"

        form_data = {"fields": [{"name": "email", "value": email}]}

        response = requests.post(url, json=form_data)

        if not response.ok:
            raise HubSpotAPICallFailed("Hubspot form submission failed")

        return response.json()
