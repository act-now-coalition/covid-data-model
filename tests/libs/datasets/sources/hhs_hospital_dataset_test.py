from datapublic.common_fields import CommonFields

from libs import pipeline
from libs.datasets import taglib
from libs.datasets.sources import can_scraper_helpers as ccd_helpers
from libs.datasets.sources import hhs_hospital_dataset
from libs.datasets.taglib import UrlStr

from tests import test_helpers
from tests.libs.datasets.sources.can_scraper_helpers_test import build_can_scraper_dataframe


def test_hhs_hospital_dataset():
    """Tests HHSHopsitalStateDataset imports adult_icu_beds_capacity
    correctly and data prior to 2020-09-01 is dropped."""
    variable = ccd_helpers.ScraperVariable(
        variable_name="adult_icu_beds_capacity",
        measurement="current",
        unit="beds",
        provider="hhs",
        common_field=CommonFields.ICU_BEDS,
    )
    source_url = UrlStr("http://foo.com")

    # Start at 2020-08-31 to make sure that the first date gets dropped.
    input_data = build_can_scraper_dataframe(
        {variable: [10, 20, 30]}, source_url=source_url, start_date="2020-08-31"
    )

    class CANScraperForTest(hhs_hospital_dataset.HHSHospitalStateDataset):
        @staticmethod
        def _get_covid_county_dataset() -> ccd_helpers.CanScraperLoader:
            return ccd_helpers.CanScraperLoader(input_data)

    ds = CANScraperForTest.make_dataset()

    # Data before 2020-09-01 should have been dropped, so we are left with [20, 30]
    icu_beds = test_helpers.TimeseriesLiteral(
        [20, 30], source=taglib.Source(type="HHSHospitalState", url=source_url)
    )
    expected_ds = test_helpers.build_default_region_dataset(
        {CommonFields.ICU_BEDS: icu_beds},
        region=pipeline.Region.from_fips("36"),
        start_date="2020-09-01",
        static={CommonFields.ICU_BEDS: 30},
    )

    test_helpers.assert_dataset_like(ds, expected_ds)


def test_hhs_hospital_dataset_non_default_start_date():
    """Tests HHSHopsitalStateDataset imports adult_icu_beds_capacity
    correctly for a state with a non-default start date (Alaska) verifying
    that data prior to its start date (2020-10-06) is dropped."""
    variable = ccd_helpers.ScraperVariable(
        variable_name="adult_icu_beds_capacity",
        measurement="current",
        unit="beds",
        provider="hhs",
        common_field=CommonFields.ICU_BEDS,
    )
    source_url = UrlStr("http://foo.com")

    # Do location Alaska and start at 2020-10-05 so the first date gets dropped.
    input_data = build_can_scraper_dataframe(
        {variable: [10, 20, 30]},
        source_url=source_url,
        start_date="2020-10-05",
        location=2,
        location_id="iso1:us#iso2:us-ak",
    )

    class CANScraperForTest(hhs_hospital_dataset.HHSHospitalStateDataset):
        @staticmethod
        def _get_covid_county_dataset() -> ccd_helpers.CanScraperLoader:
            return ccd_helpers.CanScraperLoader(input_data)

    ds = CANScraperForTest.make_dataset()

    # Data before 2020-10-05 should have been dropped, so we are left with [20, 30]
    icu_beds = test_helpers.TimeseriesLiteral(
        [20, 30], source=taglib.Source(type="HHSHospitalState", url=source_url)
    )
    expected_ds = test_helpers.build_default_region_dataset(
        {CommonFields.ICU_BEDS: icu_beds},
        region=pipeline.Region.from_fips("02"),
        start_date="2020-10-06",
        static={CommonFields.ICU_BEDS: 30},
    )

    test_helpers.assert_dataset_like(ds, expected_ds)
