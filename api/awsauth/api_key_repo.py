from typing import Optional, Dict, Any
import os

import dynamo_client


API_KEY_TABLE_NAME = os.environ["API_KEYS_TABLE"]
API_KEY_INDEX_NAME = "apiKeys"


class APIKeyRepo:
    """Logic for handling API Key functionality."""

    @staticmethod
    def add_api_key(email: str, api_key: str):
        """Adds an email and api_key to database.

        Args:
            email: Email Address.
            api_key: API Key to add.
        """
        # TODO: Client should only be initialized once
        client = dynamo_client.DynamoDBClient()
        obj = {"email": email, "api_key": api_key}
        client.put_item(API_KEY_TABLE_NAME, obj)

    @staticmethod
    def get_api_key(email) -> Optional[str]:
        """Fetches API Key for email, returning none if not found.

        Args:
            email: Email address.

        Returns: API Key if exists, None if missing.
        """
        # TODO: Client should only be initialized once
        client = dynamo_client.DynamoDBClient()

        key = {"email": email}
        api_key_item = client.get_item(API_KEY_TABLE_NAME, key)

        if not api_key_item:
            return None

        return api_key_item["api_key"]

    @staticmethod
    def get_record_for_api_key(api_key) -> Optional[Dict[str, Any]]:
        """Fetches record for API Key.

        Args:
            api_key: API Key to query.

        Returns: Dict if exists, None if not found.
        """
        # TODO: Client should only be initialized once
        client = dynamo_client.DynamoDBClient()
        items = client.query_index(API_KEY_TABLE_NAME, API_KEY_INDEX_NAME, "api_key", api_key)
        if not items:
            return None

        if len(items) > 1:
            raise Exception("Multiple emails found for API key")

        return items[0]
