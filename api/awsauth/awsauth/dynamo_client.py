from typing import Dict, List, Any, Optional

import boto3
from boto3.dynamodb.conditions import Key

from awsauth.config import Config


class DynamoDBClient:
    def __init__(self, client=None):
        self._client = client or boto3.resource("dynamodb", region_name=Config.Constants.AWS_REGION)

    def get_item(self, table: str, key: Dict[str, Any]) -> Optional[Dict[Any, Any]]:
        """Get single item from table.

        Args:
            table: Table name
            key: Key to query table with. Should be in form {'<attribute name>': <attribute value>}.

        Returns: Row if exists, None if missing.
        """
        table = self._client.Table(table)
        result = table.get_item(Key=key)

        if "Item" not in result:
            return None

        return result["Item"]

    def query_index(self, table: str, index: str, key: str, value: Any) -> List[dict]:
        """Query index for items where `key` == `value`.

        Args:
            table: Table name.
            index: Index name.
            key: Key to query.
            value: value.

        Returns: List of rows matching query.
        """
        table = self._client.Table(table)

        result = table.query(IndexName=index, KeyConditionExpression=Key(key).eq(value))
        return result["Items"]

    def put_item(self, table: str, item: Dict[str, Any]) -> None:
        """Add item to table.

        Args:
            table: Table name.
            item: Item to add in form {'<attribute_name>': <attribute_value>, ...}.

        """
        table = self._client.Table(table)
        table.put_item(Item=item)

    def update_item(self, table: str, key: dict, **updates):
        table = self._client.Table(table)
        update_expression = "set " + ", ".join(f"{column}=:{column}" for column in updates)
        table.update_item(
            Key=key,
            UpdateExpression=update_expression,
            ExpressionAttributeValues={f":{column}": value for column, value in updates.items()},
            ReturnValues="UPDATED_NEW",
        )
