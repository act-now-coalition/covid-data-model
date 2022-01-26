import io
import pandas as pd
from datapublic.common_fields import DemographicBucket

from libs.datasets import timeseries
from libs.datasets import new_cases_and_deaths
from tests import test_helpers
from libs.pipeline import Region
from datapublic.common_fields import CommonFields


def test_calculate_new_cases():
    mrts_before = timeseries.MultiRegionDataset.from_csv(
        io.StringIO(
            "location_id,date,cases\n"
            "iso1:us#iso2:us-tx#fips:48011,2020-01-01,0\n"
            "iso1:us#iso2:us-tx#fips:48011,2020-01-02,1\n"
            "iso1:us#iso2:us-tx#fips:48011,2020-01-03,1\n"
            "iso1:us#iso2:us-tx#fips:48021,2020-01-01,5\n"
            "iso1:us#iso2:us-tx#fips:48021,2020-01-02,7\n"
            "iso1:us#iso2:us-tx#fips:48031,2020-01-01,9\n"
            "iso1:us#iso2:us-tx#fips:48041,2020-01-01,\n"
            "iso1:us#iso2:us-tx#fips:48011,,100\n"
            "iso1:us#iso2:us-tx#fips:48021,,\n"
            "iso1:us#iso2:us-tx#fips:48031,,\n"
            "iso1:us#iso2:us-tx#fips:48041,,\n"
        )
    )

    mrts_expected = timeseries.MultiRegionDataset.from_csv(
        io.StringIO(
            "location_id,date,cases,new_cases\n"
            "iso1:us#iso2:us-tx#fips:48011,2020-01-01,0,\n"
            "iso1:us#iso2:us-tx#fips:48011,2020-01-02,1,1\n"
            "iso1:us#iso2:us-tx#fips:48011,2020-01-03,1,0\n"
            "iso1:us#iso2:us-tx#fips:48021,2020-01-01,5,\n"
            "iso1:us#iso2:us-tx#fips:48021,2020-01-02,7,2\n"
            "iso1:us#iso2:us-tx#fips:48031,2020-01-01,9,\n"
            "iso1:us#iso2:us-tx#fips:48041,2020-01-01,,\n"
            "iso1:us#iso2:us-tx#fips:48011,,100,0.0\n"
            "iso1:us#iso2:us-tx#fips:48021,,,2.0\n"
            "iso1:us#iso2:us-tx#fips:48031,,,\n"
            "iso1:us#iso2:us-tx#fips:48041,,,\n"
        )
    )

    timeseries_after = new_cases_and_deaths.add_new_cases(mrts_before)
    test_helpers.assert_dataset_like(mrts_expected, timeseries_after)


def test_calculate_new_deaths():
    ma_region = Region.from_state("MA")
    ny_region = Region.from_state("NY")

    dataset_before = test_helpers.build_dataset(
        {
            ma_region: {CommonFields.DEATHS: [0, 1, 1]},
            ny_region: {CommonFields.DEATHS: [None, 5, 7]},
        }
    )

    dataset_after = new_cases_and_deaths.add_new_deaths(dataset_before)
    dataset_expected = test_helpers.build_dataset(
        {
            ma_region: {CommonFields.DEATHS: [0, 1, 1], CommonFields.NEW_DEATHS: [None, 1, 0]},
            ny_region: {
                CommonFields.DEATHS: [None, 5, 7],
                CommonFields.NEW_DEATHS: [None, None, 2],
            },
        }
    )
    test_helpers.assert_dataset_like(dataset_after, dataset_expected)


def test_calculate_new_deaths_preserve_buckets():
    age_40s = DemographicBucket("age:40-49")
    metrics_before = {
        CommonFields.DEATHS: {DemographicBucket.ALL: [0, 1, 1], age_40s: [None, 5, 7]}
    }
    dataset_before = test_helpers.build_default_region_dataset(metrics_before)

    dataset_after = new_cases_and_deaths.add_new_deaths(dataset_before)
    dataset_expected = test_helpers.build_default_region_dataset(
        {
            CommonFields.NEW_DEATHS: {
                DemographicBucket.ALL: [None, 1, 0],
                age_40s: [None, None, 2],
            },
            **metrics_before,
        }
    )
    test_helpers.assert_dataset_like(dataset_after, dataset_expected)


def test_new_cases_remove_negative():
    mrts_before = timeseries.MultiRegionDataset.from_csv(
        io.StringIO(
            "location_id,date,cases\n"
            "iso1:us#fips:97111,2020-01-01,100\n"
            "iso1:us#fips:97111,2020-01-02,50\n"
            "iso1:us#fips:97111,2020-01-03,75\n"
            "iso1:us#fips:97111,2020-01-04,74\n"
            "iso1:us#fips:97111,,75\n"
        )
    )

    mrts_expected = timeseries.MultiRegionDataset.from_csv(
        io.StringIO(
            "location_id,date,cases,new_cases\n"
            "iso1:us#fips:97111,2020-01-01,100,\n"
            "iso1:us#fips:97111,2020-01-02,50,\n"
            "iso1:us#fips:97111,2020-01-03,75,25\n"
            "iso1:us#fips:97111,2020-01-04,74,0\n"
            "iso1:us#fips:97111,,75,0.0\n"
        )
    )

    timeseries_after = new_cases_and_deaths.add_new_cases(mrts_before)
    test_helpers.assert_dataset_like(mrts_expected, timeseries_after)


def test_new_cases_gap_in_date():
    mrts_before = timeseries.MultiRegionDataset.from_csv(
        io.StringIO(
            "location_id,date,cases\n"
            "iso1:us#fips:97111,2020-01-01,100\n"
            "iso1:us#fips:97111,2020-01-02,\n"
            "iso1:us#fips:97111,2020-01-03,110\n"
            "iso1:us#fips:97111,2020-01-04,130\n"
        )
    )

    mrts_expected = timeseries.MultiRegionDataset.from_csv(
        io.StringIO(
            "location_id,date,cases,new_cases\n"
            "iso1:us#fips:97111,2020-01-01,100,\n"
            "iso1:us#fips:97111,2020-01-02,,\n"
            "iso1:us#fips:97111,2020-01-03,110,\n"
            "iso1:us#fips:97111,2020-01-04,130,20\n"
        )
    )

    timeseries_after = new_cases_and_deaths.add_new_cases(mrts_before)
    test_helpers.assert_dataset_like(mrts_expected, timeseries_after)


def test_spread_first_reported_value():
    series = test_helpers.series_with_date_index([None, None, 10, 5, 0, 0, 0, 12, 5])

    results = new_cases_and_deaths.spread_first_reported_value_after_stall(series)

    expected = test_helpers.series_with_date_index([None, None, 10, 5, 3, 3, 3, 3, 5])

    pd.testing.assert_series_equal(expected, results)


def test_spread_first_reported_value_leading_zeros():
    series = test_helpers.series_with_date_index([0, 0, 10, 5, 0, 0, 0, 12, 5])

    results = new_cases_and_deaths.spread_first_reported_value_after_stall(series)

    # We don't spread across leading zeros.
    expected = test_helpers.series_with_date_index([0.0, 0, 10, 5, 3, 3, 3, 3, 5])

    pd.testing.assert_series_equal(expected, results)


def test_spread_first_reported_value_trailing_zeros():
    series = test_helpers.series_with_date_index([5, 0, 0, 0, 12, 0, 0])

    results = new_cases_and_deaths.spread_first_reported_value_after_stall(series)

    # We should leave trailing zeros intact.
    expected = test_helpers.series_with_date_index([5.0, 3, 3, 3, 3, 0, 0])

    pd.testing.assert_series_equal(expected, results)


def test_spread_first_reported_value_max_days():
    series = test_helpers.series_with_date_index([None, None, 10, 5, 0, 0, 0, 12, 5])

    results = new_cases_and_deaths.spread_first_reported_value_after_stall(
        series, max_days_to_spread=2
    )

    expected = test_helpers.series_with_date_index([None, None, 10, 5, 0, 0, 6, 6, 5])

    pd.testing.assert_series_equal(expected, results)


def test_spreading_multiple_zeros():
    series = test_helpers.series_with_date_index([None, 10, 0, 5, 0, 0, 15, 0, 20, 5])

    results = new_cases_and_deaths.spread_first_reported_value_after_stall(series)

    expected = test_helpers.series_with_date_index([None, 10, 2.5, 2.5, 5, 5, 5, 10, 10, 5])

    pd.testing.assert_series_equal(expected, results)


def test_spread_first_reported_value_empty_series():
    series = test_helpers.series_with_date_index([0])

    results = new_cases_and_deaths.spread_first_reported_value_after_stall(
        series, max_days_to_spread=2
    )

    expected = test_helpers.series_with_date_index([0])

    pd.testing.assert_series_equal(expected, results)
