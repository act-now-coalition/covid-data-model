from typing import List
import dataclasses
import pydantic
from api import can_api_v2_definition


from openapi_schema_pydantic import OpenAPI

from openapi_schema_pydantic.util import PydanticSchema
from openapi_schema_pydantic.util import construct_open_api_with_schema_class

COUNTY_TAG = "County Data"
STATE_TAG = "State Data"


fips_parameter = {
    "name": "fips",
    "in": "path",
    "required": True,
    "description": "5 Letter County FIPS code",
    "schema": {"type": "string"},
}
state_parameter = {
    "name": "state",
    "in": "path",
    "required": True,
    "description": "2 Letter State Cocde",
    "schema": {"type": "string"},
}


@dataclasses.dataclass
class APIEndpoint:

    endpoint: str

    parameters: List[dict]
    tags: List[str]
    description: str
    summary: str
    schema_cls: pydantic.BaseModel

    @property
    def endpoint_with_auth(self):
        return self.endpoint + "?apiKey={apiKey}"

    @property
    def open_api_data(self):
        security = {
            "name": "apiKey",
            "in": "query",
            "required": True,
            "schema": {"type": "string"},
        }
        parameters = self.parameters + [security]
        return {
            "parameters": parameters,
            "get": {
                "summary": self.summary,
                "description": self.description,
                "tags": self.tags,
                "responses": {
                    "200": {
                        "description": "",
                        "content": {
                            "application/json": {
                                "schema": PydanticSchema(schema_class=self.schema_cls)
                            }
                        },
                    }
                },
            },
        }


COUNTY_SUMMARY = APIEndpoint(
    endpoint="/county/{fips}.json",
    parameters=[fips_parameter],
    tags=[COUNTY_TAG],
    description="""
Region Summary object for a single county.

Lots happening with region summaries.
    """,
    summary="Single County Summary",
    schema_cls=can_api_v2_definition.RegionSummary,
)
COUNTY_TIMESERIES = APIEndpoint(
    endpoint="/county/{fips}.timeseries.json",
    parameters=[fips_parameter],
    tags=[COUNTY_TAG],
    description="Region Summary with Timeseries for a single county.",
    summary="Single County Timeseries",
    schema_cls=can_api_v2_definition.RegionSummaryWithTimeseries,
)
STATE_SUMMARY = APIEndpoint(
    endpoint="/state/{state}.json",
    parameters=[state_parameter],
    tags=[STATE_TAG],
    description="Region Summary object for a single state.",
    summary="Single State Summary",
    schema_cls=can_api_v2_definition.RegionSummary,
)
STATE_TIMESERIES = APIEndpoint(
    endpoint="/state/{state}.timeseries.json",
    parameters=[state_parameter],
    tags=[STATE_TAG],
    description="Region Summary with Timeseries for a single state.",
    summary="Single State Timeseries",
    schema_cls=can_api_v2_definition.RegionSummaryWithTimeseries,
)


ALL_STATE_SUMMARY = APIEndpoint(
    endpoint="/states.json",
    parameters=[],
    tags=[STATE_TAG],
    description="Region Summaries for all states",
    summary="All states summary",
    schema_cls=can_api_v2_definition.AggregateRegionSummary,
)
ALL_STATE_TIMESERIES = APIEndpoint(
    endpoint="/states.timeseries.json",
    parameters=[],
    tags=[STATE_TAG],
    description="Region summaries with timeseries for all states",
    summary="All states timeseries",
    schema_cls=can_api_v2_definition.AggregateRegionSummaryWithTimeseries,
)

ALL_COUNTY_SUMMARY = APIEndpoint(
    endpoint="/counties.json",
    parameters=[],
    tags=[COUNTY_TAG],
    description="Region Summaries for all counties",
    summary="All counties summary (json)",
    schema_cls=can_api_v2_definition.AggregateRegionSummary,
)
ALL_COUNTY_SUMMARY_CSV = APIEndpoint(
    endpoint="/counties.csv",
    parameters=[],
    tags=[COUNTY_TAG],
    description="Region Summaries for all counties",
    summary="All counties summary (csv)",
    schema_cls=can_api_v2_definition.AggregateFlattenedTimeseries,
)
ALL_COUNTY_TIMESERIES = APIEndpoint(
    endpoint="/counties.timeseries.json",
    parameters=[],
    tags=[COUNTY_TAG],
    description="Region summaries with timeseries for all counties",
    summary="All counties timeseries",
    schema_cls=can_api_v2_definition.AggregateRegionSummaryWithTimeseries,
)


ALL_ENDPOINTS = [
    COUNTY_SUMMARY,
    COUNTY_TIMESERIES,
    STATE_SUMMARY,
    STATE_TIMESERIES,
    ALL_STATE_SUMMARY,
    ALL_STATE_TIMESERIES,
    ALL_COUNTY_SUMMARY,
    ALL_COUNTY_SUMMARY_CSV,
    ALL_COUNTY_TIMESERIES,
]


def construct_open_api_spec() -> OpenAPI:
    api_description = """
The Covid Act Now API provides historical covid projections updated daily.
"""

    api_key_description = """
An API key is required.

Register for an API key [here](/getting-started/access).
    """
    spec = OpenAPI.parse_obj(
        {
            "info": {
                "title": "Covid Act Now API",
                "version": "v2.0.0-beta.1",
                "description": api_description,
            },
            "tags": [
                {"name": COUNTY_TAG, "description": "County level data for all US counties."},
                {
                    "name": STATE_TAG,
                    "description": "State level data for all US states + Puerto Rico and Northern Mariana Islands.",
                },
            ],
            "servers": [
                {"url": "https://api.covidactnow.org/v2", "description": "Latest available data",}
            ],
            "paths": {
                endpoint.endpoint_with_auth: endpoint.open_api_data for endpoint in ALL_ENDPOINTS
            },
            "components": {
                "securitySchemes": {
                    "API Key": {
                        "type": "apiKey",
                        "in": "query",
                        "name": "apiKey",
                        "description": api_key_description,
                    }
                }
            },
        }
    )
    return construct_open_api_with_schema_class(spec)
