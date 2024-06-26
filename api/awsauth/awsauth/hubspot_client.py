from typing import Optional
import requests
from awsauth.config import Config


class HubSpotAPICallFailed(Exception):
    """An issue with a HubSpot API call occurred."""


class HubSpotClient:
    def submit_reg_form(
        self,
        email: str,
        hubspot_token: Optional[str] = None,
        page_uri: Optional[str] = None,
        use_case: Optional[str] = None,
    ) -> Optional[dict]:
        """Submit hubspot registration form.

        Args:
            email: Email of user creating account.
            hubspot_token: Hubspot token passed from browser cookie.
            page_uri: URI of page where reg submission happened.
            use_case: Use case from form

        Returns: Successful form submission response if hubspot enabled. None if hubspot disable.

        Raises: HubSpotAPICallFailed on Hubspot error.
        """
        if not Config.Constants.HUBSPOT_ENABLED:
            return

        portal_id = Config.Constants.HUBSPOT_PORTAL_ID
        form_guid = Config.Constants.HUBSPOT_REG_FORM_GUID

        # https://legacydocs.hubspot.com/docs/methods/forms/submit_form
        url = f"https://api.hsforms.com/submissions/v3/integration/submit/{portal_id}/{form_guid}"

        fields = [{"name": "email", "value": email}]
        if use_case:
            fields.append({"name": "use_case", "value": use_case})
        form_data = {"fields": fields}

        context = {}

        if hubspot_token:
            context["hutk"] = hubspot_token
        if page_uri:
            context["pageUri"] = page_uri

        if context:
            form_data["context"] = context

        response = requests.post(url, json=form_data)

        if not response.ok:
            raise HubSpotAPICallFailed("Hubspot form submission failed")

        return response.json()
