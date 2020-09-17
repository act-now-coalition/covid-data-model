from api import can_api_v2_definition


from openapi_schema_pydantic import OpenAPI, PathItem, Operation
from openapi_schema_pydantic.util import PydanticSchema, construct_open_api_with_schema_class


def construct_base_open_api() -> OpenAPI:
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
    return OpenAPI.parse_obj(
        {
            "info": {"title": "Covid Act Now API", "version": "v0.0.1"},
            "servers": [
                {
                    "url": "https://data.covidactnow.org/v2/latest",
                    "description": "Latest available data",
                }
            ],
            "paths": {
                "/county/{fips}.timeseries.json": {
                    "parameters": [fips_parameter],
                    "get": {
                        "summary": "Single County Timeseries",
                        "description": "Region Summary with Timeseries objects for a single county.",
                        "responses": {
                            "200": {
                                "description": "Timeseries data for a single county.",
                                "content": {
                                    "application/json": {
                                        "schema": PydanticSchema(
                                            schema_class=can_api_v2_definition.AggregateRegionSummaryWithTimeseries
                                        )
                                    }
                                },
                            }
                        },
                    },
                },
                "/county/{fips}.json": {
                    "parameters": [fips_parameter],
                    "get": {
                        "summary": "Single County Summary",
                        "description": "Region Summary object for a single county.",
                        "responses": {
                            "200": {
                                "description": "Summary data for a single county.",
                                "content": {
                                    "application/json": {
                                        "schema": PydanticSchema(
                                            schema_class=can_api_v2_definition.AggregateRegionSummary
                                        )
                                    }
                                },
                            }
                        },
                    },
                },
                "/state/{state}.timeseries.json": {
                    "parameters": [state_parameter],
                    "get": {
                        "summary": "Single State Timeseries",
                        "description": "Region Summary with Timeseries objects for a single state.",
                        "responses": {
                            "200": {
                                "description": "Timeseries data for a single state.",
                                "content": {
                                    "application/json": {
                                        "schema": PydanticSchema(
                                            schema_class=can_api_v2_definition.AggregateRegionSummaryWithTimeseries
                                        )
                                    }
                                },
                            }
                        },
                    },
                },
                "/state/{state}.json": {
                    "parameters": [state_parameter],
                    "get": {
                        "summary": "Single State Summary",
                        "description": "Region Summary object for a single state.",
                        "responses": {
                            "200": {
                                "description": "Summary data for a single state.",
                                "content": {
                                    "application/json": {
                                        "schema": PydanticSchema(
                                            schema_class=can_api_v2_definition.AggregateRegionSummary
                                        )
                                    }
                                },
                            }
                        },
                    },
                },
            },
        }
    )
