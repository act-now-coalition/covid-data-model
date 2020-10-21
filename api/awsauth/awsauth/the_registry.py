import inspect
from awsauth.ses_client import SESClient
from awsauth.firehose_client import FirehoseClient
from awsauth.dynamo_client import DynamoDBClient
from awsauth.ses_client import SESClient
from awsauth.hubspot_client import HubSpotClient


class RegistryProvider:
    @property
    def ses_client(self) -> SESClient:
        return SESClient()

    @property
    def firehose_client(self) -> FirehoseClient:
        return FirehoseClient()

    @property
    def dynamodb_client(self) -> DynamoDBClient:
        return DynamoDBClient()

    @property
    def hubspot_client(self) -> HubSpotClient:
        return HubSpotClient()


class Registry(object):
    def __init__(self):
        self._initialized_objects = {}

    def initialize(self):
        properties = [
            prop for prop, obj in inspect.getmembers(self.__class__) if isinstance(obj, property)
        ]
        provider = RegistryProvider()

        for prop in properties:
            self._initialized_objects[prop] = getattr(provider, prop)

    @property
    def ses_client(self) -> SESClient:
        return self._initialized_objects["ses_client"]

    @property
    def firehose_client(self) -> FirehoseClient:
        return self._initialized_objects["firehose_client"]

    @property
    def dynamodb_client(self) -> DynamoDBClient:
        return self._initialized_objects["dynamodb_client"]

    @property
    def hubspot_client(self) -> HubSpotClient:
        return self._initialized_objects["hubspot_client"]


# `registry` is a global object that contains connections to external resources.
# It is a singleton and should only be instantiated once.
registry = Registry()
