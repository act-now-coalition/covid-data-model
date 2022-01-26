"""Helpers to access and query data loaded from the CAN Scraper parquet file.
"""
from typing import List
import enum
import dataclasses
from typing import Optional
from typing import Tuple
import requests
import more_itertools
import structlog
import pandas as pd
from datapublic.common_fields import FieldNameAndCommonField
from datapublic.common_fields import GetByValueMixin
from datapublic.common_fields import CommonFields
from datapublic.common_fields import PdFields

from libs.dataclass_utils import dataclass_with_default_init
from libs.datasets import dataset_utils
from libs.datasets import taglib
from libs.datasets.demographics import DistributionBucket


# Airflow jobs output a single parquet file with all of the data - this is where
# it is currently stored.
GCS_PARQUET_PATH = (
    "https://storage.googleapis.com/can-scrape-outputs/final/can_scrape_api_covid_us.parquet"
)

LOCAL_PARQUET_PATH = dataset_utils.DATA_DIRECTORY / "can_scrape_api_covid_us.parquet"


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
    return param.apply(lambda v: f"{v:0>{1 if v == 0 else 2 if v < 100 else 5}}")


DEMOGRAPHIC_FIELDS = [Fields.AGE, Fields.RACE, Fields.ETHNICITY, Fields.SEX]


def make_short_name(row: pd.Series) -> str:
    """Transform a Series of demographic values into a single string such as 'age;sex:0-9;female'"""
    distribution_bucket = DistributionBucket.make_from_row(DEMOGRAPHIC_FIELDS, row)
    return str(distribution_bucket)


class BadLocationId(UserWarning):
    pass


@dataclass_with_default_init(frozen=True)
class CanScraperLoader:

    all_df: pd.DataFrame
    indexed_df: pd.DataFrame

    # noinspection PyMissingConstructor
    def __init__(self, all_df: pd.DataFrame):
        # Always pre-populate indexed_df property. When it was a cached_property there were
        # mysterious consistent occurrences of
        #   worker 'gw1' crashed while running
        #   'tests/libs/datasets/data_source_test.py::test_state_providers_smoke_test'
        self.__default_init__(  # pylint: disable=E1101
            all_df=all_df, indexed_df=CanScraperLoader._make_indexed_df(all_df)
        )

    @staticmethod
    def _make_indexed_df(all_df: pd.DataFrame) -> pd.DataFrame:
        """The parquet file with many fields moved into a MultiIndex and demographic fields
        transformed into a single string."""
        # Make a Series of bucket short names with a MultiIndex of the unique DEMOGRAPHIC_FIELDS.
        # There are only a few (~50) unique fields among the millions of rows. `apply`ing
        # make_short_name to every row takes much longer than using a join to copy from this series.
        # This pattern of finding unique values and copying with a join is also used in
        # taglib.Series.attribute_df_to_json_series.
        bucket_short_names = (
            all_df.loc[:, DEMOGRAPHIC_FIELDS]
            .drop_duplicates()
            .set_index(DEMOGRAPHIC_FIELDS, drop=False)
            .apply(make_short_name, axis=1, result_type="reduce")
            .rename(PdFields.DEMOGRAPHIC_BUCKET)
        )
        CanScraperLoader._check_location_id(all_df)
        # Use `join(other, on=...)` because it preserves the indexed_df index
        rv = (
            all_df.join(bucket_short_names, on=DEMOGRAPHIC_FIELDS)
            .drop(columns=DEMOGRAPHIC_FIELDS)
            .set_index(
                [
                    Fields.PROVIDER,
                    Fields.VARIABLE_NAME,
                    Fields.MEASUREMENT,
                    Fields.UNIT,
                    Fields.LOCATION_ID,
                    PdFields.DEMOGRAPHIC_BUCKET,
                    Fields.DATE,
                ]
            )
        )
        return rv

    @staticmethod
    def _check_location_id(all_df: pd.DataFrame):
        # TODO(tom): Maybe merge this function and timeseries._map_and_warn_about_mismatches. They
        #  are similar but the inputs are slightly different so can't be bothered to refactor now.
        location_id_fips_df = all_df.loc[:, [Fields.LOCATION_ID, Fields.LOCATION]].drop_duplicates()

        COMPUTED_LOCATION_ID = "computed_location_id"
        location_id_fips_df[COMPUTED_LOCATION_ID] = _fips_from_int(
            location_id_fips_df[Fields.LOCATION]
        ).map(dataset_utils.get_fips_to_location())

        bad_location_id_mask = (
            location_id_fips_df[Fields.LOCATION_ID] != location_id_fips_df[COMPUTED_LOCATION_ID]
        )
        if bad_location_id_mask.any():
            bad_location_ids = location_id_fips_df.loc[
                bad_location_id_mask, [Fields.LOCATION_ID, COMPUTED_LOCATION_ID]
            ]
            bad_rows = bad_location_ids.merge(all_df, how="left", on=Fields.LOCATION_ID)
            raise BadLocationId(
                f"Bad location_id. Examples:\n" f"{bad_rows.to_string(line_width=200, max_rows=20)}"
            )

    def query_multiple_variables(
        self,
        variables: List[ScraperVariable],
        *,
        log_provider_coverage_warnings: bool = False,
        source_type: str,
    ) -> Tuple[pd.Series, pd.DataFrame]:
        """Queries multiple variables

        Args:
            variables: Variables to query
            log_provider_coverage_warnings: Log warnings when upstream data has variables not in
              `variables` and hints when a variable has no data.
            source_type: String for the `taglib.Source.type` property

        Returns:
            The observations in a Series and the sources in a tag DataFrame
        """
        if log_provider_coverage_warnings:
            self.check_variable_coverage(variables)

        # Split `variables` into lists of variables dropped and those returned.
        variables_to_drop = [variable for variable in variables if not variable.common_field]
        variables_to_return = [variable for variable in variables if variable.common_field]

        for v in variables_to_drop:
            # Verify agreement with ScraperVariable docstring.
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
        for v in variables_to_return:
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

        unknown_columns = set(combined_rows.columns) - set(Fields) - {PdFields.DEMOGRAPHIC_BUCKET}
        if unknown_columns:
            raise ValueError(f"Unknown column. Add {unknown_columns} to Fields.")

        indexed_rows = (
            combined_rows.rename_axis(
                index={Fields.DATE: CommonFields.DATE, Fields.LOCATION_ID: CommonFields.LOCATION_ID}
            )
            .reorder_levels(
                [
                    CommonFields.LOCATION_ID,
                    PdFields.VARIABLE,
                    PdFields.DEMOGRAPHIC_BUCKET,
                    CommonFields.DATE,
                ]
            )
            .sort_index()
        )

        dups = indexed_rows.index.duplicated(keep=False)
        if dups.any():
            raise NotImplementedError(
                f"No support for aggregating duplicate observations:\n"
                f"{indexed_rows.loc[dups].to_string(line_width=200, max_rows=200, max_colwidth=40)}"
            )

        # For now only making a source tag for observations with bucket "all".
        tag_df = taglib.Source.rename_and_make_tag_df(
            indexed_rows.xs("all", axis=0, level=PdFields.DEMOGRAPHIC_BUCKET, drop_level=True),
            source_type=source_type,
            rename={Fields.SOURCE_URL: "url", Fields.SOURCE_NAME: "name"},
        )

        rows = indexed_rows[PdFields.VALUE]
        return rows, tag_df

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
    def persist_parquet():
        """Fetches the up-to-date Parquet file from GCS and stores it in data/"""
        payload = requests.get(GCS_PARQUET_PATH).content
        with (LOCAL_PARQUET_PATH) as file:
            file.write_bytes(payload)

    @staticmethod
    def load_from_local() -> "CanScraperLoader":
        """Returns a CanScraperLoader which holds data loaded from the CAN Scraper."""

        all_df = pd.read_parquet(LOCAL_PARQUET_PATH)

        return CanScraperLoader(all_df)
