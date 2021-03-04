import io
from libs.datasets import timeseries
from libs.datasets import new_cases_and_deaths
from tests import test_helpers


def test_calculate_new_cases():
    mrts_before = timeseries.MultiRegionDataset.from_csv(
        io.StringIO(
            "location_id,date,cases\n"
            "iso1:us#fips:1,2020-01-01,0\n"
            "iso1:us#fips:1,2020-01-02,1\n"
            "iso1:us#fips:1,2020-01-03,1\n"
            "iso1:us#fips:2,2020-01-01,5\n"
            "iso1:us#fips:2,2020-01-02,7\n"
            "iso1:us#fips:3,2020-01-01,9\n"
            "iso1:us#fips:4,2020-01-01,\n"
            "iso1:us#fips:1,,100\n"
            "iso1:us#fips:2,,\n"
            "iso1:us#fips:3,,\n"
            "iso1:us#fips:4,,\n"
        )
    )

    mrts_expected = timeseries.MultiRegionDataset.from_csv(
        io.StringIO(
            "location_id,date,cases,new_cases\n"
            "iso1:us#fips:1,2020-01-01,0,0\n"
            "iso1:us#fips:1,2020-01-02,1,1\n"
            "iso1:us#fips:1,2020-01-03,1,0\n"
            "iso1:us#fips:2,2020-01-01,5,5\n"
            "iso1:us#fips:2,2020-01-02,7,2\n"
            "iso1:us#fips:3,2020-01-01,9,9\n"
            "iso1:us#fips:4,2020-01-01,,\n"
            "iso1:us#fips:1,,100,0.0\n"
            "iso1:us#fips:2,,,2.0\n"
            "iso1:us#fips:3,,,9.0\n"
            "iso1:us#fips:4,,,\n"
        )
    )

    timeseries_after = new_cases_and_deaths.add_new_cases(mrts_before)
    test_helpers.assert_dataset_like(mrts_expected, timeseries_after)


def test_calculate_new_deaths():
    mrts_before = timeseries.MultiRegionDataset.from_csv(
        io.StringIO(
            "location_id,date,deaths\n"
            "iso1:us#fips:1,2020-01-01,0\n"
            "iso1:us#fips:1,2020-01-02,1\n"
            "iso1:us#fips:1,2020-01-03,1\n"
            "iso1:us#fips:2,2020-01-01,5\n"
            "iso1:us#fips:2,2020-01-02,7\n"
            "iso1:us#fips:3,2020-01-01,9\n"
            "iso1:us#fips:4,2020-01-01,\n"
            "iso1:us#fips:1,,100\n"
            "iso1:us#fips:2,,\n"
            "iso1:us#fips:3,,\n"
            "iso1:us#fips:4,,\n"
        )
    )

    mrts_expected = timeseries.MultiRegionDataset.from_csv(
        io.StringIO(
            "location_id,date,deaths,new_deaths\n"
            "iso1:us#fips:1,2020-01-01,0,0\n"
            "iso1:us#fips:1,2020-01-02,1,1\n"
            "iso1:us#fips:1,2020-01-03,1,0\n"
            "iso1:us#fips:2,2020-01-01,5,5\n"
            "iso1:us#fips:2,2020-01-02,7,2\n"
            "iso1:us#fips:3,2020-01-01,9,9\n"
            "iso1:us#fips:4,2020-01-01,,\n"
            "iso1:us#fips:1,,100,0.0\n"
            "iso1:us#fips:2,,,2.0\n"
            "iso1:us#fips:3,,,9.0\n"
            "iso1:us#fips:4,,,\n"
        )
    )

    timeseries_after = new_cases_and_deaths.add_new_deaths(mrts_before)
    test_helpers.assert_dataset_like(mrts_expected, timeseries_after)


def test_new_cases_remove_negative():
    mrts_before = timeseries.MultiRegionDataset.from_csv(
        io.StringIO(
            "location_id,date,cases\n"
            "iso1:us#fips:1,2020-01-01,100\n"
            "iso1:us#fips:1,2020-01-02,50\n"
            "iso1:us#fips:1,2020-01-03,75\n"
            "iso1:us#fips:1,2020-01-04,74\n"
            "iso1:us#fips:1,,75\n"
        )
    )

    mrts_expected = timeseries.MultiRegionDataset.from_csv(
        io.StringIO(
            "location_id,date,cases,new_cases\n"
            "iso1:us#fips:1,2020-01-01,100,100\n"
            "iso1:us#fips:1,2020-01-02,50,\n"
            "iso1:us#fips:1,2020-01-03,75,25\n"
            "iso1:us#fips:1,2020-01-04,74,0\n"
            "iso1:us#fips:1,,75,0.0\n"
        )
    )

    timeseries_after = new_cases_and_deaths.add_new_cases(mrts_before)
    test_helpers.assert_dataset_like(mrts_expected, timeseries_after)


def test_new_cases_gap_in_date():
    mrts_before = timeseries.MultiRegionDataset.from_csv(
        io.StringIO(
            "location_id,date,cases\n"
            "iso1:us#fips:1,2020-01-01,100\n"
            "iso1:us#fips:1,2020-01-02,\n"
            "iso1:us#fips:1,2020-01-03,110\n"
            "iso1:us#fips:1,2020-01-04,130\n"
        )
    )

    mrts_expected = timeseries.MultiRegionDataset.from_csv(
        io.StringIO(
            "location_id,date,cases,new_cases\n"
            "iso1:us#fips:1,2020-01-01,100,100\n"
            "iso1:us#fips:1,2020-01-02,,\n"
            "iso1:us#fips:1,2020-01-03,110,\n"
            "iso1:us#fips:1,2020-01-04,130,20\n"
        )
    )

    timeseries_after = new_cases_and_deaths.add_new_cases(mrts_before)
    test_helpers.assert_dataset_like(mrts_expected, timeseries_after)
