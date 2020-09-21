from typing import Dict, Any
import os
import re
import uuid
import json
import logging
import boto3
from boto3.dynamodb.conditions import Key

_logger = logging.getLogger(__name__)

API_KEY_TABLE_NAME = os.environ["API_KEYS_TABLE"]
API_KEY_INDEX_NAME = "apiKeys"


class DynamoDBClient:
    def __init__(self, client=None):
        self._client = client or boto3.resource("dynamodb")

    def get_item(self, table, key):
        table = self._client.Table(table)
        result = table.get_item(Key=key)

        if "Item" not in result:
            return None

        return result["Item"]

    def query_index(self, table, index, key, value):
        table = self._client.Table(table)

        result = table.query(IndexName=index, KeyConditionExpression=Key(key).eq(value))
        return result["Items"]

    def put_item(self, table: str, item: Dict[str, Any]):
        table = self._client.Table(table)
        return table.put_item(Item=item)


class APIKeyRepo:
    @staticmethod
    def add_api_key(email, api_key):
        client = DynamoDBClient()
        obj = {"email": email, "api_key": api_key}
        client.put_item(API_KEY_TABLE_NAME, obj)

    @staticmethod
    def get_api_key(email):
        client = DynamoDBClient()

        key = {"email": email}
        api_key_item = client.get_item(API_KEY_TABLE_NAME, key)

        if not api_key_item:
            return None

        return api_key_item["api_key"]

    @staticmethod
    def get_record_for_api_key(api_key):
        client = DynamoDBClient()
        items = client.query_index(API_KEY_TABLE_NAME, API_KEY_INDEX_NAME, "api_key", api_key)
        if not items:
            return None

        if len(items) > 1:
            raise Exception("Multiple emails found for API key")

        return items[0]


class InvalidAPIKey(Exception):
    def __init__(self, api_key):
        super().__init__(f"Invalid API Key: {api_key}")
        self.api_key = api_key


def _create_api_key(email: str) -> str:
    return uuid.uuid4().hex


def _get_or_create_api_key(email):
    api_key = APIKeyRepo.get_api_key(email)
    if api_key:
        return api_key

    _logger.info(f"No API Key found for email {email}, creating new key")

    api_key = _create_api_key(email)
    APIKeyRepo.add_api_key(email, api_key)

    return api_key


def _check_api_key(api_key):
    if not APIKeyRepo.get_record_for_api_key(api_key):
        raise InvalidAPIKey(api_key)


def register(event, context):

    if not "email" in event:
        raise ValueError("Missing email parameter")

    email = event["email"]
    api_key = _get_or_create_api_key(email)
    body = {"api_key": api_key, "email": email}

    response = {"statusCode": 200, "body": json.dumps(body)}

    return response

    # Use this code if you don't use the http event with the LAMBDA-PROXY
    # integration
    """
    return {
        "message": "Go Serverless v1.0! Your function executed successfully!",
        "event": event
    }
    """


def check_api_key(event, context):
    if not event["queryStringParameters"]["apiKey"]:
        raise Exception("Must have api key")
    api_key = event["queryStringParameters"]["apiKey"]
    record = APIKeyRepo.get_record_for_api_key(api_key)
    if not record:
        raise InvalidAPIKey(api_key)

    principalId = record["email"]

    tmp = event["methodArn"].split(":")
    apiGatewayArnTmp = tmp[5].split("/")
    awsAccountId = tmp[4]
    policy = AuthPolicy(principalId, awsAccountId)
    policy.restApiId = apiGatewayArnTmp[0]
    policy.region = tmp[3]
    policy.stage = apiGatewayArnTmp[1]
    policy.allowAllMethods()

    return policy.build()

    # event.get('queryStringParameters')
    # body = {
    #     "message": "Go Serverless v1.0! Your function executed successfully!",
    #     "input": event
    # }
    response = {"statusCode": 200, "body": json.dumps(body)}

    return response


class HttpVerb:
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    PATCH = "PATCH"
    HEAD = "HEAD"
    DELETE = "DELETE"
    OPTIONS = "OPTIONS"
    ALL = "*"


class AuthPolicy(object):
    awsAccountId = ""
    """The AWS account id the policy will be generated for. This is used to create the method ARNs."""
    principalId = ""
    """The principal used for the policy, this should be a unique identifier for the end user."""
    version = "2012-10-17"
    """The policy version used for the evaluation. This should always be '2012-10-17'"""
    pathRegex = "^[/.a-zA-Z0-9-\*]+$"
    """The regular expression used to validate resource paths for the policy"""

    """these are the internal lists of allowed and denied methods. These are lists
    of objects and each object has 2 properties: A resource ARN and a nullable
    conditions statement.
    the build method processes these lists and generates the approriate
    statements for the final policy"""
    allowMethods = []
    denyMethods = []

    restApiId = "*"
    """The API Gateway API id. By default this is set to '*'"""
    region = "*"
    """The region where the API is deployed. By default this is set to '*'"""
    stage = "*"
    """The name of the stage used in the policy. By default this is set to '*'"""

    def __init__(self, principal, awsAccountId):
        self.awsAccountId = awsAccountId
        self.principalId = principal
        self.allowMethods = []
        self.denyMethods = []

    def _addMethod(self, effect, verb, resource, conditions):
        """Adds a method to the internal lists of allowed or denied methods. Each object in
        the internal list contains a resource ARN and a condition statement. The condition
        statement can be null."""
        if verb != "*" and not hasattr(HttpVerb, verb):
            raise NameError("Invalid HTTP verb " + verb + ". Allowed verbs in HttpVerb class")
        resourcePattern = re.compile(self.pathRegex)
        if not resourcePattern.match(resource):
            raise NameError(
                "Invalid resource path: " + resource + ". Path should match " + self.pathRegex
            )

        if resource[:1] == "/":
            resource = resource[1:]

        resourceArn = (
            "arn:aws:execute-api:"
            + self.region
            + ":"
            + self.awsAccountId
            + ":"
            + self.restApiId
            + "/"
            + self.stage
            + "/"
            + verb
            + "/"
            + resource
        )

        if effect.lower() == "allow":
            self.allowMethods.append({"resourceArn": resourceArn, "conditions": conditions})
        elif effect.lower() == "deny":
            self.denyMethods.append({"resourceArn": resourceArn, "conditions": conditions})

    def _getEmptyStatement(self, effect):
        """Returns an empty statement object prepopulated with the correct action and the
        desired effect."""
        statement = {
            "Action": "execute-api:Invoke",
            "Effect": effect[:1].upper() + effect[1:].lower(),
            "Resource": [],
        }

        return statement

    def _getStatementForEffect(self, effect, methods):
        """This function loops over an array of objects containing a resourceArn and
        conditions statement and generates the array of statements for the policy."""
        statements = []

        if len(methods) > 0:
            statement = self._getEmptyStatement(effect)

            for curMethod in methods:
                if curMethod["conditions"] is None or len(curMethod["conditions"]) == 0:
                    statement["Resource"].append(curMethod["resourceArn"])
                else:
                    conditionalStatement = self._getEmptyStatement(effect)
                    conditionalStatement["Resource"].append(curMethod["resourceArn"])
                    conditionalStatement["Condition"] = curMethod["conditions"]
                    statements.append(conditionalStatement)

            statements.append(statement)

        return statements

    def allowAllMethods(self):
        """Adds a '*' allow to the policy to authorize access to all methods of an API"""
        self._addMethod("Allow", HttpVerb.ALL, "*", [])

    def denyAllMethods(self):
        """Adds a '*' allow to the policy to deny access to all methods of an API"""
        self._addMethod("Deny", HttpVerb.ALL, "*", [])

    def allowMethod(self, verb, resource):
        """Adds an API Gateway method (Http verb + Resource path) to the list of allowed
        methods for the policy"""
        self._addMethod("Allow", verb, resource, [])

    def denyMethod(self, verb, resource):
        """Adds an API Gateway method (Http verb + Resource path) to the list of denied
        methods for the policy"""
        self._addMethod("Deny", verb, resource, [])

    def allowMethodWithConditions(self, verb, resource, conditions):
        """Adds an API Gateway method (Http verb + Resource path) to the list of allowed
        methods and includes a condition for the policy statement. More on AWS policy
        conditions here: http://docs.aws.amazon.com/IAM/latest/UserGuide/reference_policies_elements.html#Condition"""
        self._addMethod("Allow", verb, resource, conditions)

    def denyMethodWithConditions(self, verb, resource, conditions):
        """Adds an API Gateway method (Http verb + Resource path) to the list of denied
        methods and includes a condition for the policy statement. More on AWS policy
        conditions here: http://docs.aws.amazon.com/IAM/latest/UserGuide/reference_policies_elements.html#Condition"""
        self._addMethod("Deny", verb, resource, conditions)

    def build(self):
        """Generates the policy document based on the internal lists of allowed and denied
        conditions. This will generate a policy with two main statements for the effect:
        one statement for Allow and one statement for Deny.
        Methods that includes conditions will have their own statement in the policy."""
        if (self.allowMethods is None or len(self.allowMethods) == 0) and (
            self.denyMethods is None or len(self.denyMethods) == 0
        ):
            raise NameError("No statements defined for the policy")

        policy = {
            "principalId": self.principalId,
            "policyDocument": {"Version": self.version, "Statement": []},
        }

        policy["policyDocument"]["Statement"].extend(
            self._getStatementForEffect("Allow", self.allowMethods)
        )
        policy["policyDocument"]["Statement"].extend(
            self._getStatementForEffect("Deny", self.denyMethods)
        )

        return policy
