import dataclasses
import itertools
from typing import Dict, List
import io
import datetime
from typing import Iterable
from typing import Iterator
from typing import Optional
from typing import Union

import pytest
from covidactnow.datapublic.common_fields import CommonFields
from covidactnow.datapublic import common_df
import pandas as pd
from covidactnow.datapublic.common_fields import PdFields

from libs.datasets import taglib
from libs.datasets.sources import can_scraper_helpers as ccd_helpers


# Match fields in the CAN Scraper DB
from libs.datasets.taglib import UrlStr

DEFAULT_LOCATION = "36"
DEFAULT_LOCATION_TYPE = "state"


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
    start_date="2021-01-01",
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
    results, _ = data.query_multiple_variables([variable], source_type="MySource")

    expected_buf = io.StringIO(
        "fips,      date,vaccinations_completed\n"
        f" 36,2021-01-01,                    10\n"
        f" 36,2021-01-02,                    20\n"
        f" 36,2021-01-03,                    30\n".replace(" ", "")
    )
    expected = common_df.read_csv(expected_buf, set_index=False)
    pd.testing.assert_frame_equal(expected, results, check_names=False)


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
    data = ccd_helpers.CanScraperLoader(input_data)
    results, _ = data.query_multiple_variables([variable], source_type="MySource")

    expected_buf = io.StringIO(
        "fips,      date,cases\n"
        f" 36,2021-01-01,  100\n"
        f" 36,2021-01-02,  100\n".replace(" ", "")
    )
    expected = common_df.read_csv(expected_buf, set_index=False)
    pd.testing.assert_frame_equal(expected, results, check_names=False)


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
    data = ccd_helpers.CanScraperLoader(input_data)
    results, tags = data.query_multiple_variables([variable], source_type="MySource")

    expected_data_buf = io.StringIO(
        "fips,      date,vaccinations_completed\n"
        "  36,2021-01-01,                    10\n"
        "  36,2021-01-02,                    20\n"
        "  36,2021-01-03,                    30\n".replace(" ", "")
    )
    expected = common_df.read_csv(expected_data_buf, set_index=False)
    pd.testing.assert_frame_equal(expected, results, check_names=False)

    expected_tag_df = pd.DataFrame(
        {
            CommonFields.FIPS: expected[CommonFields.FIPS],
            CommonFields.DATE: expected[CommonFields.DATE],
            PdFields.VARIABLE: "vaccinations_completed",
            taglib.TagField.TYPE: taglib.TagType.SOURCE,
            taglib.TagField.CONTENT: taglib.Source(type="MySource", url=source_url).content,
        }
    )
    pd.testing.assert_frame_equal(expected_tag_df, tags, check_like=True, check_names=False)


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
