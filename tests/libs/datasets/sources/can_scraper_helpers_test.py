from typing import Dict, List
import io
import datetime
from covidactnow.datapublic.common_fields import CommonFields
from covidactnow.datapublic import common_df
import pandas as pd

from libs.datasets.sources import can_scraper_helpers as ccd_helpers


# Match fields in the CAN Scraper DB
DEFAULT_LOCATION = "36"
DEFAULT_LOCATION_TYPE = "state"


def _build_can_scraper_dataframe(
    data_by_variable: Dict[ccd_helpers.ScraperVariable, List[float]],
    location=DEFAULT_LOCATION,
    location_type=DEFAULT_LOCATION_TYPE,
    start_date="2021-01-01",
) -> pd.DataFrame:
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
                "sex": variable.sex,
                "value": value,
            }
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

    input_data = _build_can_scraper_dataframe(
        {variable: [10, 20, 30], not_included_variable: [10, 20, 40]}
    )
    data = ccd_helpers.CanScraperLoader(input_data)
    results = data.query_multiple_variables([variable])

    expected_buf = io.StringIO(
        "fips,date,aggregate_level,vaccinations_completed\n"
        f"36,2021-01-01,state,10\n"
        f"36,2021-01-02,state,20\n"
        f"36,2021-01-03,state,30\n"
    )
    expected = common_df.read_csv(expected_buf, set_index=False)
    pd.testing.assert_frame_equal(expected, results)
