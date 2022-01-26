import numpy as np
from datapublic.common_fields import CommonFields

from tests import test_helpers
from tests.test_helpers import TimeseriesLiteral
from libs.datasets import nytimes_anomalies
from libs.datasets.taglib import TagType
from libs.pipeline import Region


def test_read_nytimes_anomalies():
    anomalies = nytimes_anomalies.read_nytimes_anomalies()
    assert anomalies.size > 0


def test_filter_nytimes_anomalies_expand_subgeographies():
    # Test that this anomaly applies to WI counties but not the state.
    # 2020-09-04,,,Wisconsin,USA-55,cases,,yes,Wisconsin began reporting probable cases at the county level.

    region_wi = Region.from_state("WI")
    region_milwaukee = Region.from_fips("55079")

    ds_in = test_helpers.build_dataset(
        {
            region_wi: {CommonFields.NEW_CASES: [1.0, 2.0, 3.0]},
            region_milwaukee: {CommonFields.NEW_CASES: [4.0, 5.0, 6.0]},
        },
        start_date="2020-09-03",
    )

    expected_tag = test_helpers.make_tag(
        TagType.KNOWN_ISSUE,
        date="2020-09-04",
        public_note="NYTimes: Wisconsin began reporting probable cases at the county level.",
    )
    ds_expected = test_helpers.build_dataset(
        {
            region_wi: {CommonFields.NEW_CASES: [1.0, 2.0, 3.0]},
            region_milwaukee: {
                CommonFields.NEW_CASES: TimeseriesLiteral(
                    [4.0, np.nan, 6.0], annotation=[expected_tag]
                )
            },
        },
        start_date="2020-09-03",
    )

    ds_out = nytimes_anomalies.filter_by_nyt_anomalies(ds_in)
    test_helpers.assert_dataset_like(ds_out, ds_expected, drop_na_dates=True)


def test_filter_nytimes_anomalies_deaths_expand_dates():
    # Test that this anomaly applies to OH dates within range.
    # 2021-03-05,2021-03-08,,Ohio,USA-39,deaths,yes,,Ohio added more than 400 deaths of residents who died out of state.

    region_oh = Region.from_state("OH")

    ds_in = test_helpers.build_dataset(
        {region_oh: {CommonFields.NEW_DEATHS: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]},},
        start_date="2021-03-04",
    )

    expected_tags = [
        test_helpers.make_tag(
            TagType.KNOWN_ISSUE,
            date=date,
            public_note="NYTimes: Ohio added more than 400 deaths of residents who died out of state.",
        )
        for date in ["2021-03-05", "2021-03-06", "2021-03-07", "2021-03-08"]
    ]

    ds_expected = test_helpers.build_dataset(
        {
            region_oh: {
                CommonFields.NEW_DEATHS: TimeseriesLiteral(
                    [1.0, np.nan, np.nan, np.nan, np.nan, 6.0], annotation=expected_tags
                )
            }
        },
        start_date="2021-03-04",
    )

    ds_out = nytimes_anomalies.filter_by_nyt_anomalies(ds_in)
    test_helpers.assert_dataset_like(ds_out, ds_expected, drop_na_dates=True)


def test_filter_nytimes_anomalies_deaths_expand_both_to_cases_and_deaths():
    # Test that this anomaly applies to both cases and deaths in MI.
    # 2020-06-01,,,Michigan,USA-26,both,yes,,The Times began including probable cases and deaths reported by Michigan's county and regional health districts.

    region_mi = Region.from_state("MI")

    ds_in = test_helpers.build_dataset(
        {
            region_mi: {
                CommonFields.NEW_CASES: [1.0, 2.0, 3.0],
                CommonFields.NEW_DEATHS: [4.0, 5.0, 6.0],
            }
        },
        start_date="2020-05-31",
    )

    expected_tag = test_helpers.make_tag(
        TagType.KNOWN_ISSUE,
        date="2020-06-01",
        public_note="NYTimes: The Times began including probable cases and deaths reported by Michigan's county and regional health districts.",
    )

    ds_expected = test_helpers.build_dataset(
        {
            region_mi: {
                CommonFields.NEW_CASES: TimeseriesLiteral(
                    [1.0, np.nan, 3.0], annotation=[expected_tag]
                ),
                CommonFields.NEW_DEATHS: TimeseriesLiteral(
                    [4.0, np.nan, 6.0], annotation=[expected_tag]
                ),
            }
        },
        start_date="2020-05-31",
    )

    ds_out = nytimes_anomalies.filter_by_nyt_anomalies(ds_in)
    test_helpers.assert_dataset_like(ds_out, ds_expected, drop_na_dates=True)
