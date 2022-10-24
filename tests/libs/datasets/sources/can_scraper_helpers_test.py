import dataclasses
import itertools
from typing import Dict, List
import datetime
from typing import Iterable
from typing import Iterator
from typing import Optional
from typing import Union

import pytest
from datapublic.common_fields import CommonFields
import pandas as pd
from datapublic.common_fields import DemographicBucket

from libs.datasets import data_source
from libs.datasets import taglib
from libs.datasets.sources import can_scraper_helpers as ccd_helpers

from libs.datasets.taglib import UrlStr
from libs.pipeline import Region
from tests import test_helpers
from tests.test_helpers import TimeseriesLiteral


# Match fields in the CAN Scraper DB
DEFAULT_LOCATION = 36  # FIPS is an int in the parquet file, not a str
DEFAULT_LOCATION_TYPE = "state"
DEFAULT_LOCATION_ID = Region.from_fips(str(DEFAULT_LOCATION)).location_id
DEFAULT_START_DATE = test_helpers.DEFAULT_START_DATE


def _make_iterator(maybe_iterable: Union[None, str, Iterable[str]]) -> Optional[Iterator[str]]:
    if maybe_iterable is None:
        return None
    elif isinstance(maybe_iterable, str):
        return itertools.repeat(maybe_iterable)
    else:
        return iter(maybe_iterable)


def build_can_scraper_dataframe(
    data_by_variable: Dict[ccd_helpers.ScraperVariable, List[float]],
    location=DEFAULT_LOCATION,
    location_type=DEFAULT_LOCATION_TYPE,
    location_id=DEFAULT_LOCATION_ID,
    start_date=DEFAULT_START_DATE,
    source_url: Union[None, str, Iterable[str]] = None,
    source_name: Union[None, str, Iterable[str]] = None,
) -> pd.DataFrame:
    """Creates a DataFrame with the same structure as the CAN Scraper parquet file.

    Args:
        source_url: None to not include the column or a string for every observation or
        an iterable of strings to add to each observation in the order created.
    """
    source_url_iter = _make_iterator(source_url)
    source_name_iter = _make_iterator(source_name)
    start_date = datetime.datetime.fromisoformat(start_date)
    rows = []
    for variable, data in data_by_variable.items():

        for i, value in enumerate(data):
            date = start_date + datetime.timedelta(days=i)
            row = {
                "provider": variable.provider,
                "dt": date,
                "location_type": location_type,
                "location_id": location_id,
                "location": location,
                "variable_name": variable.variable_name,
                "measurement": variable.measurement,
                "unit": variable.unit,
                "age": variable.age,
                "race": variable.race,
                "ethnicity": variable.ethnicity,
                "sex": variable.sex,
                "value": value,
            }
            if source_url:
                row["source_url"] = next(source_url_iter)
            if source_name:
                row["source_name"] = next(source_name_iter)
            rows.append(row)

    return pd.DataFrame(rows)


def test_query_multiple_variables():
    variable = ccd_helpers.ScraperVariable(
        variable_name="total_vaccine_completed",
        measurement="cumulative",
        unit="people",
        provider="cdc",
        common_field=CommonFields.VACCINATIONS_COMPLETED,
    )
    not_included_variable = ccd_helpers.ScraperVariable(
        variable_name="total_vaccine_completed",
        measurement="cumulative",
        unit="people",
        # Different provider, so query shouldn't return it
        provider="hhs",
        common_field=CommonFields.VACCINATIONS_COMPLETED,
    )

    input_data = build_can_scraper_dataframe(
        {variable: [10, 20, 30], not_included_variable: [10, 20, 40]}
    )
    data = ccd_helpers.CanScraperLoader(input_data)

    # Make a new subclass to keep this test separate from others in the make_dataset lru_cache.
    class CANScraperForTest(data_source.CanScraperBase):
        VARIABLES = [variable]
        SOURCE_TYPE = "MySource"

        @staticmethod
        def _get_covid_county_dataset() -> ccd_helpers.CanScraperLoader:
            return data

    ds = CANScraperForTest.make_dataset()

    vaccinations_completed = TimeseriesLiteral([10, 20, 30], source=taglib.Source(type="MySource"))
    expected_ds = test_helpers.build_default_region_dataset(
        {CommonFields.VACCINATIONS_COMPLETED: vaccinations_completed},
        region=Region.from_fips("36"),
    )

    test_helpers.assert_dataset_like(ds, expected_ds)


def test_query_multiple_variables_with_ethnicity():
    variable = ccd_helpers.ScraperVariable(
        variable_name="cases",
        measurement="cumulative",
        unit="people",
        provider="cdc",
        common_field=CommonFields.CASES,
        ethnicity="all",
    )
    variable_hispanic = dataclasses.replace(variable, ethnicity="hispanic")

    input_data = build_can_scraper_dataframe({variable: [100, 100], variable_hispanic: [40, 40]})

    class CANScraperForTest(data_source.CanScraperBase):
        VARIABLES = [variable]
        SOURCE_TYPE = "MySource"

        @staticmethod
        def _get_covid_county_dataset() -> ccd_helpers.CanScraperLoader:
            return ccd_helpers.CanScraperLoader(input_data)

    ds = CANScraperForTest.make_dataset()

    cases = TimeseriesLiteral([100, 100], source=taglib.Source(type="MySource"))
    expected_ds = test_helpers.build_default_region_dataset(
        {
            CommonFields.CASES: {
                # bucket="all" has a taglib.Source with it; other buckets have only the numbers
                DemographicBucket.ALL: cases,
                DemographicBucket("ethnicity:hispanic"): [40, 40],
            }
        },
        region=Region.from_fips("36"),
    )

    test_helpers.assert_dataset_like(ds, expected_ds)


def test_query_source_url():
    variable = ccd_helpers.ScraperVariable(
        variable_name="total_vaccine_completed",
        measurement="cumulative",
        unit="people",
        provider="cdc",
        common_field=CommonFields.VACCINATIONS_COMPLETED,
    )
    source_url = UrlStr("http://foo.com")
    input_data = build_can_scraper_dataframe({variable: [10, 20, 30]}, source_url=source_url)

    class CANScraperForTest(data_source.CanScraperBase):
        VARIABLES = [variable]
        SOURCE_TYPE = "MySource"

        @staticmethod
        def _get_covid_county_dataset() -> ccd_helpers.CanScraperLoader:
            return ccd_helpers.CanScraperLoader(input_data)

    ds = CANScraperForTest.make_dataset()

    vaccinations_completed = TimeseriesLiteral(
        [10, 20, 30], source=taglib.Source(type="MySource", url=source_url)
    )
    expected_ds = test_helpers.build_default_region_dataset(
        {CommonFields.VACCINATIONS_COMPLETED: vaccinations_completed},
        region=Region.from_fips("36"),
    )

    test_helpers.assert_dataset_like(ds, expected_ds)


def test_query_multiple_variables_extra_field():
    variable = ccd_helpers.ScraperVariable(
        variable_name="cases",
        measurement="cumulative",
        unit="people",
        provider="cdc",
        common_field=CommonFields.CASES,
    )
    input_data = build_can_scraper_dataframe({variable: [10, 20, 30]})
    input_data["extra_column"] = 123
    data = ccd_helpers.CanScraperLoader(input_data)
    with pytest.raises(ValueError):
        data.query_multiple_variables([variable], source_type="MySource")


# TODO(michael): Reenable.
@pytest.mark.skip(reason="Temporary hack to allow / remove duplicates.")
def test_query_multiple_variables_duplicate_observation():
    variable = ccd_helpers.ScraperVariable(
        variable_name="cases",
        measurement="cumulative",
        unit="people",
        provider="cdc",
        common_field=CommonFields.CASES,
        ethnicity="all",
    )

    input_data = build_can_scraper_dataframe({variable: [100, 100]})
    data = ccd_helpers.CanScraperLoader(pd.concat([input_data, input_data]))
    with pytest.raises(NotImplementedError):
        data.query_multiple_variables([variable], source_type="MySource")


def test_bad_location_id():
    variable = ccd_helpers.ScraperVariable(
        variable_name="cases",
        measurement="cumulative",
        unit="people",
        provider="cdc",
        common_field=CommonFields.CASES,
    )
    input_data = build_can_scraper_dataframe({variable: [10, 20, 30]})
    input_data.iat[0, input_data.columns.get_loc(ccd_helpers.Fields.LOCATION_ID)] = "iso1:us#nope"
    with pytest.raises(ccd_helpers.BadLocationId, match=r"\#nope"):
        ccd_helpers.CanScraperLoader(input_data)
