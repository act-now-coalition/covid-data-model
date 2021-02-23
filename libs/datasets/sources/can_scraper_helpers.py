"""Helpers to access and query data loaded from the CAN Scraper parquet file.
"""
from typing import List
import enum
import dataclasses
from typing import Optional
from typing import Tuple

import more_itertools
import structlog
import pandas as pd
from covidactnow.datapublic.common_fields import FieldNameAndCommonField
from covidactnow.datapublic.common_fields import GetByValueMixin
from covidactnow.datapublic.common_fields import CommonFields


# Airflow jobs output a single parquet file with all of the data - this is where
# it is currently stored.
from covidactnow.datapublic.common_fields import PdFields

from libs.datasets import dataset_utils
from libs.datasets import taglib

PARQUET_PATH = "data/can-scrape/can_scrape_api_covid_us.parquet"


_logger = structlog.getLogger()


@enum.unique
class Fields(GetByValueMixin, FieldNameAndCommonField, enum.Enum):
    """Fields in CCD DataFrame."""

    PROVIDER = "provider", None
    DATE = "dt", CommonFields.DATE
    LOCATION = "location", CommonFields.FIPS
    LOCATION_ID = "location_id", None
    LOCATION_TYPE = "location_type", None
    VARIABLE_NAME = "variable_name", PdFields.VARIABLE
    MEASUREMENT = "measurement", None
    UNIT = "unit", None
    AGE = "age", None
    RACE = "race", None
    ETHNICITY = "ethnicity", None
    SEX = "sex", None
    VALUE = "value", PdFields.VALUE
    SOURCE_URL = "source_url", None
    SOURCE_NAME = "source_name", None
    LAST_UPDATED = "last_updated", None


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
    ethnicity: str = "all"
    sex: str = "all"


def _fips_from_int(param: pd.Series):
    """Transform FIPS from an int64 to a string of 2 or 5 chars.

    See https://github.com/valorumdata/covid_county_data.py/issues/3

    Copied from covid-data-public/scripts/helpers.py
    """
    return param.apply(lambda v: f"{v:0>{2 if v < 100 else 5}}")


@dataclasses.dataclass(frozen=True)
class CanScraperLoader:

    all_df: pd.DataFrame

    def _get_rows(self, variable: ScraperVariable) -> pd.DataFrame:
        # Similar to dataset_utils.make_rows_key this builds a Pandas.eval query string by making a
        # list of query parts, then joining them with `and`. For our current data this takes 23s
        # while the previous method of making a binary mask for each variable took 54s. This is run
        # during tests so the speed up is nice to have.
        assert variable.race == "all"
        required_fields = ["provider", "variable_name", "age", "race", "ethnicity", "sex"]
        assert all([getattr(variable, field) for field in required_fields])
        query_parts = [f"{field} == @variable.{field}" for field in required_fields]
        for optional_field in ["measurement", "unit"]:
            if getattr(variable, optional_field):
                query_parts.append(f"{optional_field} == @variable.{optional_field}")

        return self.all_df.loc[self.all_df.eval(" and ".join(query_parts))].copy()

    def query_multiple_variables(
        self, variables: List[ScraperVariable], *, log_provider_coverage_warnings: bool = False
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Queries multiple variables returning wide df with variable names as columns.

        Args:
            variables: Variables to query
            log_provider_coverage_warnings: Log warnings when upstream data has variables not in
              `variables` and hints when a variable has no data.

        Returns:
            The observations in a DataFrame with variable columns and the source_urls
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
            # Copy CommonField name to data. The loop is continued above when common_field is None.
            data.loc[:, Fields.VARIABLE_NAME] = variable.common_field

            selected_data.append(data)

        # TODO(tom): check LOCATION_TYPE matches LOCATION

        combined_data = pd.concat(selected_data)
        unknown_columns = set(combined_data.columns) - set(Fields)
        if unknown_columns:
            raise ValueError(f"Unknown column. Add {unknown_columns} to Fields.")

        rename_columns = {f: f.common_field for f in Fields if f.common_field}
        long_df = combined_data.rename(columns=rename_columns)

        if Fields.SOURCE_URL in long_df.columns:
            combined_source_urls = long_df.loc[
                :, [CommonFields.FIPS, CommonFields.DATE, PdFields.VARIABLE, Fields.SOURCE_URL]
            ].rename(columns={Fields.SOURCE_URL: taglib.TagField.CONTENT})
        else:
            combined_source_urls = pd.DataFrame([])

        indexed = long_df.set_index(
            [CommonFields.FIPS, PdFields.VARIABLE, CommonFields.DATE]
        ).sort_index()
        dups = indexed.index.duplicated(keep=False)

        if dups.any():
            raise NotImplementedError(
                f"No support for aggregating duplicate observations:\n"
                f"{indexed.loc[dups].to_string(line_width=200, max_rows=200,max_colwidth=40)}"
            )

        # Make a DataFrame with index=[FIPS,DATE], column=VARIABLE and value=VALUE. This used to
        # use pivot_table but we don't want to aggregate observations. I tried using
        # `DataFrame.pivot` but couldn't work around a mysterious
        # "NotImplementedError: > 1 ndim Categorical are not supported at this time".
        wide_vars_df = indexed[PdFields.VALUE].unstack(level=PdFields.VARIABLE).reset_index()
        assert wide_vars_df.columns.names == [PdFields.VARIABLE]
        return wide_vars_df, combined_source_urls

    def check_variable_coverage(self, variables: List[ScraperVariable]):
        provider_name = more_itertools.one(set(v.provider for v in variables))
        provider_mask = self.all_df[Fields.PROVIDER] == provider_name
        counts = self.all_df.loc[provider_mask, Fields.VARIABLE_NAME].value_counts()
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
