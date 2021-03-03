from typing import List
import pydantic
import pytest
from libs.pipelines import csv_column_ordering
from api.can_api_v2_definition import RegionSummary
from api.can_api_v2_definition import RegionTimeseriesRowWithHeader


def _build_schema_names(schema: pydantic.BaseModel) -> List[str]:
    all_possible_names = []
    for key, value in schema.__fields__.items():
        all_possible_names.append(key)

        if isinstance(value.type_, pydantic.BaseModel.__class__):
            sub_names = _build_schema_names(value.type_)

            all_possible_names.extend([f"{key}.{name}" for name in sub_names])

    return all_possible_names


# Verify that columns in column ordering match schema names.
@pytest.mark.parametrize(
    "schema,csv_columns",
    [
        (RegionSummary, csv_column_ordering.SUMMARY_ORDER),
        (RegionTimeseriesRowWithHeader, csv_column_ordering.TIMESERIES_ORDER),
    ],
)
def test_csv_columns_match(schema, csv_columns):
    columns = set(csv_columns)
    possible_names = _build_schema_names(schema)
    assert not columns.difference(possible_names)
