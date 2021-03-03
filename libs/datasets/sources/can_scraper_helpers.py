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
from backports.cached_property import cached_property
from covidactnow.datapublic.common_fields import FieldNameAndCommonField
from covidactnow.datapublic.common_fields import GetByValueMixin
from covidactnow.datapublic.common_fields import CommonFields


# Airflow jobs output a single parquet file with all of the data - this is where
# it is currently stored.
from covidactnow.datapublic.common_fields import PdFields

from libs.datasets import dataset_utils
from libs.datasets import taglib

DEMOGRAPHIC_BUCKET = "demographic_bucket"

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


demo_columns = ["age", "race", "ethnicity", "sex"]


def make_short_name(row: pd.Series) -> str:
    non_all = []
    for field in demo_columns:
        if row[field] != "all":
            non_all.append(f"{field}:{row[field]}")
    if non_all:
        return ";".join(non_all)
    else:
        return "all"


@dataclasses.dataclass(frozen=True)
class CanScraperLoader:

    all_df: pd.DataFrame

    @cached_property
    def indexed_df(self) -> pd.DataFrame:
        indexed_df = self.all_df.set_index(
            [
                Fields.PROVIDER,
                Fields.VARIABLE_NAME,
                Fields.MEASUREMENT,
                Fields.UNIT,
                Fields.LOCATION_ID,
                Fields.AGE,
                Fields.RACE,
                Fields.ETHNICITY,
                Fields.SEX,
                Fields.DATE,
            ]
        ).sort_index()
        all_ares = (
            self.all_df.loc[:, demo_columns]
            .drop_duplicates()
            .set_index(demo_columns, drop=False)
            .apply(make_short_name, axis=1)
            .rename(DEMOGRAPHIC_BUCKET)
        )
        # Use `join(other, on=...)` because it preserves the indexed_df index
        rv = (
            indexed_df.join(all_ares, on=demo_columns)
            .set_index(DEMOGRAPHIC_BUCKET, append=True)
            .droplevel(demo_columns)
        )
        return rv

    def query_multiple_variables(
        self,
        variables: List[ScraperVariable],
        *,
        log_provider_coverage_warnings: bool = False,
        source_type: str,
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

        variables_drop, variables_return = [], []
        for v in variables:
            if v.common_field is None:
                variables_drop.append(v)
            else:
                variables_return.append(v)

        # Check that `variable` agrees with stuff in the ScraperVariable docstring.
        for v in variables_drop:
            assert v.measurement == ""
            assert v.unit == ""

        selected_data = {}
        indexed_df = self.indexed_df
        assert indexed_df.index.names[0:4] == [
            Fields.PROVIDER,
            Fields.VARIABLE_NAME,
            Fields.MEASUREMENT,
            Fields.UNIT,
        ]
        #                                  Fields.LOCATION_ID, Fields.DATE, DEMOGRAPHIC_BUCKET]
        for v in variables_return:
            # Must be set when copying to the return value
            assert v.measurement
            assert v.unit

            try:
                selected_data[v.common_field] = indexed_df.loc(axis=0)[
                    v.provider, v.variable_name, v.measurement, v.unit
                ]
            except KeyError:
                pass

        combined_rows = pd.concat(selected_data, axis=0, names=[PdFields.VARIABLE])

        unknown_columns = set(combined_rows.columns) - set(Fields) - {DEMOGRAPHIC_BUCKET}
        if unknown_columns:
            raise ValueError(f"Unknown column. Add {unknown_columns} to Fields.")

        indexed = (
            combined_rows.rename_axis(
                index={Fields.DATE: CommonFields.DATE, Fields.LOCATION_ID: CommonFields.LOCATION_ID}
            )
            .reorder_levels(
                [CommonFields.LOCATION_ID, PdFields.VARIABLE, DEMOGRAPHIC_BUCKET, CommonFields.DATE]
            )
            .sort_index()
        )

        dups = indexed.index.duplicated(keep=False)
        if dups.any():
            raise NotImplementedError(
                f"No support for aggregating duplicate observations:\n"
                f"{indexed.loc[dups].to_string(line_width=200, max_rows=200,max_colwidth=40)}"
            )

        tag_df = taglib.Source.rename_and_make_tag_df(
            indexed.xs("all", axis=0, level=DEMOGRAPHIC_BUCKET, drop_level=True),
            source_type=source_type,
            rename={Fields.SOURCE_URL: "url", Fields.SOURCE_NAME: "name"},
        )

        rows_df = indexed[PdFields.VALUE].xs(
            "all", axis=0, level=DEMOGRAPHIC_BUCKET, drop_level=True
        )
        return rows_df, tag_df

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
