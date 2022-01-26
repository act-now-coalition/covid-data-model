from typing import List
import dataclasses
import pydantic
from api import can_api_v2_definition


from openapi_schema_pydantic import OpenAPI

from openapi_schema_pydantic.util import PydanticSchema
from openapi_schema_pydantic.util import construct_open_api_with_schema_class

COUNTY_TAG = "County Data"
STATE_TAG = "State Data"
CBSA_TAG = "CBSA Data"
COUNTRY_TAG = "Country Data"


fips_parameter = {
    "name": "fips",
    "in": "path",
    "required": True,
    "description": "5 Letter County FIPS code",
    "schema": {"type": "string"},
}
cbsa_parameter = {
    "name": "cbsa_code",
    "in": "path",
    "required": True,
    "description": """
5 Letter core-based statistical area (CBSA) Code.

For a list of all CBSA codes, refer to the
[Census.gov Delineation Files](https://www.census.gov/geographies/reference-files/time-series/demo/metro-micro/delineation-files.html)""",
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
        content_type = "application/csv" if self.endpoint.endswith(".csv") else "application/json"
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
                            content_type: {"schema": PydanticSchema(schema_class=self.schema_cls)}
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
    schema_cls=can_api_v2_definition.AggregateRegionSummary,
)
ALL_COUNTY_TIMESERIES = APIEndpoint(
    endpoint="/counties.timeseries.json",
    parameters=[],
    tags=[COUNTY_TAG],
    description="Region summaries with timeseries for all counties",
    summary="All counties timeseries",
    schema_cls=can_api_v2_definition.AggregateRegionSummaryWithTimeseries,
)
ALL_COUNTY_TIMESERIES_CSV = APIEndpoint(
    endpoint="/counties.timeseries.csv",
    parameters=[],
    tags=[COUNTY_TAG],
    description="Region summaries with timeseries for all counties",
    summary="All counties timeseries (csv)",
    schema_cls=can_api_v2_definition.AggregateFlattenedTimeseries,
)

# All bulk files for counties in a specific state
STATE_COUNTY_SUMMARY = APIEndpoint(
    endpoint="/county/{state}.json",
    parameters=[state_parameter],
    tags=[COUNTY_TAG],
    description="Region summaries for all counties in a state",
    summary="Counties summary in state (json)",
    schema_cls=can_api_v2_definition.AggregateRegionSummary,
)
STATE_COUNTY_SUMMARY_CSV = APIEndpoint(
    endpoint="/county/{state}.csv",
    parameters=[state_parameter],
    tags=[COUNTY_TAG],
    description="Region summaries for all counties in a state",
    summary="Counties summary in state (csv)",
    schema_cls=can_api_v2_definition.AggregateRegionSummary,
)
STATE_COUNTY_TIMESERIES = APIEndpoint(
    endpoint="/county/{state}.timeseries.json",
    parameters=[state_parameter],
    tags=[COUNTY_TAG],
    description="Region summaries with timeseries for all counties in a state",
    summary="Counties timeseries in state",
    schema_cls=can_api_v2_definition.AggregateRegionSummaryWithTimeseries,
)
STATE_COUNTY_TIMESERIES_CSV = APIEndpoint(
    endpoint="/county/{state}.timeseries.csv",
    parameters=[state_parameter],
    tags=[COUNTY_TAG],
    description="Region summaries with timeseries for all counties in a state",
    summary="Counties timeseries in state (csv)",
    schema_cls=can_api_v2_definition.AggregateFlattenedTimeseries,
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
STATE_TIMESERIES_CSV = APIEndpoint(
    endpoint="/state/{state}.timeseries.csv",
    parameters=[state_parameter],
    tags=[STATE_TAG],
    description="Region Summary with Timeseries for a single state.",
    summary="Single State Timeseries CSV",
    schema_cls=can_api_v2_definition.AggregateRegionSummaryWithTimeseries,
)
ALL_STATE_SUMMARY = APIEndpoint(
    endpoint="/states.json",
    parameters=[],
    tags=[STATE_TAG],
    description="Region Summaries for all states",
    summary="All states summary (json)",
    schema_cls=can_api_v2_definition.AggregateRegionSummary,
)
ALL_STATE_SUMMARY_CSV = APIEndpoint(
    endpoint="/states.csv",
    parameters=[],
    tags=[STATE_TAG],
    description="Region Summaries for all states",
    summary="All states summary (csv)",
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
ALL_STATE_TIMESERIES_CSV = APIEndpoint(
    endpoint="/states.timeseries.csv",
    parameters=[],
    tags=[STATE_TAG],
    description="Region summaries with timeseries for all states",
    summary="All states timeseries (csv)",
    schema_cls=can_api_v2_definition.AggregateFlattenedTimeseries,
)

CBSA_SUMMARY = APIEndpoint(
    endpoint="/cbsa/{cbsa_code}.json",
    parameters=[cbsa_parameter],
    tags=[CBSA_TAG],
    description="Region Summary object for a single CBSA.",
    summary="Single CBSA Summary",
    schema_cls=can_api_v2_definition.RegionSummary,
)
CBSA_TIMESERIES = APIEndpoint(
    endpoint="/cbsa/{cbsa_code}.timeseries.json",
    parameters=[cbsa_parameter],
    tags=[CBSA_TAG],
    description="Region Summary with Timeseries for a single CBSA.",
    summary="Single CBSA Timeseries",
    schema_cls=can_api_v2_definition.RegionSummaryWithTimeseries,
)
ALL_CBSA_SUMMARY = APIEndpoint(
    endpoint="/cbsas.json",
    parameters=[],
    tags=[CBSA_TAG],
    description="Region Summaries for all CBSAs",
    summary="All CBSAs summary (json)",
    schema_cls=can_api_v2_definition.AggregateRegionSummary,
)
ALL_CBSA_SUMMARY_CSV = APIEndpoint(
    endpoint="/cbsas.csv",
    parameters=[],
    tags=[CBSA_TAG],
    description="Region Summaries for all CBSAs",
    summary="All CBSAs summary (csv)",
    schema_cls=can_api_v2_definition.AggregateRegionSummary,
)
ALL_CBSA_TIMESERIES = APIEndpoint(
    endpoint="/cbsas.timeseries.json",
    parameters=[],
    tags=[CBSA_TAG],
    description="Region summaries with timeseries for all CBSAs",
    summary="All CBSAs timeseries",
    schema_cls=can_api_v2_definition.AggregateRegionSummaryWithTimeseries,
)
ALL_CBSA_TIMESERIES_CSV = APIEndpoint(
    endpoint="/cbsas.timeseries.csv",
    parameters=[],
    tags=[CBSA_TAG],
    description="Region summaries with timeseries for all CBSAs",
    summary="All CBSAs timeseries (csv)",
    schema_cls=can_api_v2_definition.AggregateFlattenedTimeseries,
)


US_SUMMARY = APIEndpoint(
    endpoint="/country/US.json",
    parameters=[],
    tags=[COUNTRY_TAG],
    description="Region Summary object for US.",
    summary="US Summary",
    schema_cls=can_api_v2_definition.RegionSummary,
)
US_TIMESERIES = APIEndpoint(
    endpoint="/country/US.timeseries.json",
    parameters=[],
    tags=[COUNTRY_TAG],
    description="Region Summary with Timeseries for US.",
    summary="US Timeseries",
    schema_cls=can_api_v2_definition.RegionSummaryWithTimeseries,
)
US_TIMESERIES_CSV = APIEndpoint(
    endpoint="/country/US.timeseries.csv",
    parameters=[],
    tags=[COUNTRY_TAG],
    description="Region Summary with Timeseries for US.",
    summary="US Timeseries CSV",
    schema_cls=can_api_v2_definition.AggregateRegionSummaryWithTimeseries,
)


ALL_ENDPOINTS = [
    COUNTY_SUMMARY,
    COUNTY_TIMESERIES,
    STATE_SUMMARY,
    STATE_TIMESERIES,
    STATE_TIMESERIES_CSV,
    CBSA_SUMMARY,
    CBSA_TIMESERIES,
    US_SUMMARY,
    US_TIMESERIES,
    US_TIMESERIES_CSV,
    ALL_STATE_SUMMARY,
    ALL_STATE_SUMMARY_CSV,
    ALL_STATE_TIMESERIES,
    ALL_STATE_TIMESERIES_CSV,
    ALL_COUNTY_SUMMARY,
    ALL_COUNTY_SUMMARY_CSV,
    ALL_COUNTY_TIMESERIES,
    ALL_COUNTY_TIMESERIES_CSV,
    STATE_COUNTY_SUMMARY,
    STATE_COUNTY_SUMMARY_CSV,
    STATE_COUNTY_TIMESERIES,
    STATE_COUNTY_TIMESERIES_CSV,
    ALL_CBSA_SUMMARY,
    ALL_CBSA_SUMMARY_CSV,
    ALL_CBSA_TIMESERIES,
    ALL_CBSA_TIMESERIES_CSV,
]


CBSA_DESCRIPTION = """
Aggregated data for all [core-based statistical areas
(CBSA)](https://en.wikipedia.org/wiki/Core-based_statistical_area).

CBSAs represent collections of counties that are socioeconomically linked.

They are used to represent metropolitan and micropolitan (at least 10,000 people
and fewer than 50,000 people) areas.

For example, the Seattle-Tacoma-Bellevue, WA CBSA is an aggregation of King County,
Pierce County, and Snohomish County.

CBSAs are currently in beta and may not contain all metrics or data.
"""


def build_model_tag(schema):
    schema_name = schema.__name__
    return {
        "name": schema_name,
        "description": f'<SchemaDefinition schemaRef="#/components/schemas/{schema_name}" showReadOnly={{true}} showWriteOnly={{true}} />',
    }


MODEL_TAGS = [
    build_model_tag(can_api_v2_definition.Actuals),
    build_model_tag(can_api_v2_definition.Metrics),
    build_model_tag(can_api_v2_definition.RiskLevels),
    build_model_tag(can_api_v2_definition.RegionSummary),
    build_model_tag(can_api_v2_definition.RegionSummaryWithTimeseries),
]


def construct_open_api_spec() -> dict:
    api_description = """
The Covid Act Now API provides historical covid projections updated daily.
"""

    api_key_description = """
An API key is required.

Register for an API key [here](/#register).
    """
    spec = OpenAPI.parse_obj(
        {
            "info": {
                "title": "Covid Act Now API",
                "version": "v2.0.0",
                "description": api_description,
            },
            "tags": [
                {"name": COUNTY_TAG, "description": "County level data for all US counties."},
                {
                    "name": STATE_TAG,
                    "description": "State level data for all US states + Puerto Rico and Northern Mariana Islands.",
                },
                {"name": CBSA_TAG, "description": CBSA_DESCRIPTION},
                *MODEL_TAGS,
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
    open_api_schema = construct_open_api_with_schema_class(spec)
    spec = open_api_schema.dict(by_alias=True, exclude_none=True)

    # x-tagGroups are a vendored extension for redocly and are not in the
    # `openapi_schema_pydantic` pydantic classes, so they have to be manually
    # added after the schema is built.
    spec["x-tagGroups"] = [
        {"name": "Endpoints", "tags": [STATE_TAG, COUNTY_TAG, CBSA_TAG, COUNTRY_TAG]},
        {"name": "Models", "tags": [tag["name"] for tag in MODEL_TAGS]},
    ]
    return spec
