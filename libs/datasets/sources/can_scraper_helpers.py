"""Helpers to access and query data loaded from the CAN Scraper parquet file.
"""
from typing import List
import enum
import dataclasses
from typing import Optional

import more_itertools
import structlog
import pandas as pd
from covidactnow.datapublic.common_fields import FieldNameAndCommonField
from covidactnow.datapublic.common_fields import GetByValueMixin
from covidactnow.datapublic.common_fields import CommonFields


# Airflow jobs output a single parquet file with all of the data - this is where
# it is currently stored.
from libs.datasets import dataset_utils

PARQUET_PATH = "data/can-scrape/can_scrape_api_covid_us.parquet"


_logger = structlog.getLogger()


@enum.unique
class Fields(GetByValueMixin, FieldNameAndCommonField, enum.Enum):
    """Fields in CCD DataFrame."""

    PROVIDER = "provider", None
    DATE = "dt", CommonFields.DATE
    LOCATION_TYPE = "location_type", CommonFields.AGGREGATE_LEVEL
    # Special transformation to FIPS
    LOCATION = "location", CommonFields.FIPS
    VARIABLE_NAME = "variable_name", None
    MEASUREMENT = "measurement", None
    UNIT = "unit", None
    AGE = "age", None
    RACE = "race", None
    SEX = "sex", None
    VALUE = "value", None


@dataclasses.dataclass(frozen=True)
class ScraperVariable:
    """Represents a specific variable scraped in CAN Scraper Dataset.

    ScraperVariable has two modes:
    * If `common_field`, `measurement` and `unit` are falsy the scraper variable is dropped.
    * If they are truthy values from the scraper variable are copied to the output.
    `query_multiple_variables` asserts that each ScraperVariable is in one of these modes.
    Turning these into two different classes seems like more work than it is worth right now.

    The table at https://github.com/covid-projections/can-scrapers/blob/main/can_tools/bootstrap_data/covid_variables.csv
    """

    variable_name: str
    provider: str
    measurement: str = ""
    unit: str = ""
    common_field: Optional[CommonFields] = None
    age: str = "all"
    race: str = "all"
    sex: str = "all"


def _fips_from_int(param: pd.Series):
    """Transform FIPS from an int64 to a string of 2 or 5 chars.

    See https://github.com/valorumdata/covid_county_data.py/issues/3

    Copied from covid-data-public/scripts/helpers.py
    """
    return param.apply(lambda v: f"{v:0>{2 if v < 100 else 5}}")


@dataclasses.dataclass
class CanScraperLoader:

    timeseries_df: pd.DataFrame

    def _get_rows(self, variable: ScraperVariable) -> pd.DataFrame:
        all_df = self.timeseries_df

        is_selected_data = (
            (all_df[Fields.PROVIDER] == variable.provider)
            & (all_df[Fields.VARIABLE_NAME] == variable.variable_name)
            & (all_df[Fields.AGE] == variable.age)
            & (all_df[Fields.RACE] == variable.race)
            & (all_df[Fields.SEX] == variable.sex)
        )
        if variable.measurement:
            is_selected_data = is_selected_data & (
                all_df[Fields.MEASUREMENT] == variable.measurement
            )
        if variable.unit:
            is_selected_data = is_selected_data & (all_df[Fields.UNIT] == variable.unit)

        return all_df.loc[is_selected_data, :].copy()

    def query_multiple_variables(
        self, variables: List[ScraperVariable], *, log_provider_coverage_warnings: bool = False
    ) -> pd.DataFrame:
        """Queries multiple variables returning wide df with variable names as columns.

        Args:
            variables: Variables to query
            log_provider_coverage_warnings: Log warnings when upstream data has variables not in
              `variables` and hints when a variable has no data.
        """
        if log_provider_coverage_warnings:
            self.check_variable_coverage(variables)
        selected_data = []

        for variable in variables:
            # Check that `variable` agrees with stuff in the ScraperVariable docstring.
            if variable.common_field is None:
                assert variable.measurement == ""
                assert variable.unit == ""
                continue
            else:
                # Must be set when copying to the return value
                assert variable.measurement
                assert variable.unit

            data = self._get_rows(variable)
            if data.empty and log_provider_coverage_warnings:
                _logger.info("No data rows found for variable", variable=variable)
                more_data = self._get_rows(dataclasses.replace(variable, measurement="", unit=""))
                _logger.info(
                    "Try these parameters",
                    variable_name=variable.variable_name,
                    measurement_counts=str(more_data[Fields.MEASUREMENT].value_counts().to_dict()),
                    unit_counts=str(more_data[Fields.UNIT].value_counts().to_dict()),
                )

            # Rename fields if common field name exists
            if variable.common_field:
                data.loc[:, Fields.VARIABLE_NAME] = variable.common_field

            selected_data.append(data)

        combined_df = pd.concat(selected_data)

        wide_df = combined_df.pivot_table(
            index=[Fields.LOCATION.value, Fields.DATE.value, Fields.LOCATION_TYPE.value],
            columns=Fields.VARIABLE_NAME.value,
            values=Fields.VALUE.value,
        ).reset_index()

        data = wide_df.rename(
            columns={
                Fields.LOCATION.value: CommonFields.FIPS,
                Fields.DATE.value: CommonFields.DATE,
                Fields.LOCATION_TYPE.value: CommonFields.AGGREGATE_LEVEL,
            }
        )
        data.columns.name = None
        return data

    def check_variable_coverage(self, variables: List[ScraperVariable]):
        provider_name = more_itertools.one(set(v.provider for v in variables))
        provider_mask = self.timeseries_df[Fields.PROVIDER] == provider_name
        counts = self.timeseries_df.loc[provider_mask, Fields.VARIABLE_NAME].value_counts()
        variables_by_name = {var.variable_name: var for var in variables}
        for variable_name, count in counts.iteritems():
            if variable_name not in variables_by_name:
                _logger.info(
                    "Upstream has variable not in variables list",
                    variable_name=variable_name,
                    count=count,
                )

    @staticmethod
    def load() -> "CanScraperLoader":
        """Returns a CanScraperLoader which holds data loaded from the CAN Scraper."""

        data_root = dataset_utils.LOCAL_PUBLIC_DATA_PATH
        input_path = data_root / PARQUET_PATH

        all_df = pd.read_parquet(input_path)
        all_df[Fields.LOCATION] = _fips_from_int(all_df[Fields.LOCATION])

        return CanScraperLoader(all_df)
