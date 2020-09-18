import io

import numpy as np
import pandas as pd
from covidactnow.datapublic.common_fields import CommonFields
from covidactnow.datapublic import common_df
from api import can_api_definition
from libs import top_level_metrics
from libs.datasets.timeseries import MultiRegionTimeseriesDataset
from libs.datasets.sources.can_pyseir_location_output import CANPyseirLocationOutput
from libs.datasets.timeseries import OneRegionTimeseriesDataset
from libs.datasets.timeseries import TimeseriesDataset
from libs.pipeline import Region


def _build_metrics_df(content: str) -> pd.DataFrame:
    header = (
        "date,fips,caseDensity,testPositivityRatio,contactTracerCapacityRatio,"
        "infectionRate,infectionRateCI90,icuHeadroomRatio\n"
    )
    data = io.StringIO(f"{header}\n{content}")
    return common_df.read_csv(data, set_index=False)


def _series_with_date_index(data, date: str = "2020-08-25", **series_kwargs):
    date_series = pd.date_range(date, periods=len(data), freq="D")
    return pd.Series(data, index=date_series, **series_kwargs)


def _fips_csv_to_one_region(csv_str: str, region: Region) -> OneRegionTimeseriesDataset:
    # Make a Timeseries first because it can have a FIPS column without location_id
    ts = TimeseriesDataset.load_csv(io.StringIO(csv_str))
    # from_timeseries adds the location_id column needed by get_one_region
    return MultiRegionTimeseriesDataset.from_timeseries(ts).get_one_region(region)


def test_calculate_case_density():
    """
    It should use population, smoothing and a normalizing factor to calculate case density.
    """
    cases = _series_with_date_index([0, 0, 20, 60, 120])
    pop = 100
    every_ten = 10
    smooth = 2

    density = top_level_metrics.calculate_case_density(
        cases, pop, smooth=smooth, normalize_by=every_ten
    )

    pd.testing.assert_series_equal(
        density, _series_with_date_index([np.nan, 0.0, 1.0, 3.0, 5.0], dtype="float"),
    )


def test_calculate_test_positivity():
    """
    It should use smoothed case data to calculate positive test rate.
    """

    positive_tests = _series_with_date_index([0, 1, 3, 6])
    negative_tests = _series_with_date_index([0, 0, 1, 3])
    positive_rate = top_level_metrics.calculate_test_positivity(
        positive_tests, negative_tests, smooth=2, lag_lookback=1
    )
    expected = _series_with_date_index([np.nan, 1, 0.75, 2 / 3], dtype="float64")
    pd.testing.assert_series_equal(positive_rate, expected)


def test_calculate_test_positivity_lagging():
    """
    It should return an empty series if there is missing negative case data.
    """
    positive_tests = _series_with_date_index([0, 1, 2, 4, 8])
    negative_tests = _series_with_date_index([0, 0, 1, 2, 2])
    positive_rate = top_level_metrics.calculate_test_positivity(
        positive_tests, negative_tests, smooth=2, lag_lookback=1
    )
    pd.testing.assert_series_equal(positive_rate, pd.Series([], dtype="float64"))


def test_calculate_test_positivity_extra_day():
    """
    It should return an empty series if there is missing negative case data.
    """
    positive_tests = _series_with_date_index([0, 4, np.nan])
    negative_tests = _series_with_date_index([0, 4, np.nan])
    positive_rate = top_level_metrics.calculate_test_positivity(positive_tests, negative_tests)
    expected = _series_with_date_index([np.nan, 0.5], dtype="float64")
    pd.testing.assert_series_equal(positive_rate, expected)


def test_top_level_metrics_basic():
    data = (
        "date,fips,cases,positive_tests,negative_tests,contact_tracers_count"
        ",current_icu,current_icu_total,icu_beds\n"
        "2020-08-17,36,10,10,90,1,10,20,\n"
        "2020-08-18,36,20,20,180,2,10,20,\n"
        "2020-08-19,36,,,,3,10,20,\n"
        "2020-08-20,36,40,40,360,4,10,20,\n"
    )
    timeseries = _fips_csv_to_one_region(data, Region.from_fips("36"))
    latest = {
        CommonFields.POPULATION: 100_000,
        CommonFields.FIPS: "36",
        CommonFields.STATE: "NY",
        CommonFields.ICU_TYPICAL_OCCUPANCY_RATE: 0.5,
        CommonFields.ICU_BEDS: 30,
    }
    results, _ = top_level_metrics.calculate_metrics_for_timeseries(
        timeseries, latest, None, require_recent_icu_data=False
    )

    expected = _build_metrics_df(
        "2020-08-17,36,,,,,,0.5\n"
        "2020-08-18,36,10,0.1,0.04,,,0.5\n"
        "2020-08-19,36,,0.1,,,,0.5\n"
        "2020-08-20,36,,0.1,,,,0.5\n"
    )
    pd.testing.assert_frame_equal(expected, results)


def test_top_level_metrics_no_test_positivity():
    data = (
        "date,fips,cases,positive_tests,negative_tests,contact_tracers_count,current_icu,icu_beds\n"
        "2020-08-17,36,10,,,1,,\n"
        "2020-08-18,36,20,,,2,,\n"
        "2020-08-19,36,30,,,3,,\n"
        "2020-08-20,36,40,,,4,,\n"
    )
    timeseries = _fips_csv_to_one_region(data, Region.from_fips("36"))
    latest = {
        CommonFields.POPULATION: 100_000,
        CommonFields.FIPS: "36",
        CommonFields.STATE: "NY",
        CommonFields.ICU_BEDS: 10,
    }
    results, _ = top_level_metrics.calculate_metrics_for_timeseries(timeseries, latest, None)

    expected = _build_metrics_df(
        "2020-08-17,36,,,,,\n"
        "2020-08-18,36,10,,0.04,,\n"
        "2020-08-19,36,10,,0.06,,\n"
        "2020-08-20,36,10,,0.08,,\n"
    )
    pd.testing.assert_frame_equal(expected, results)


def test_top_level_metrics_with_rt():
    data = (
        "date,fips,cases,positive_tests,negative_tests,contact_tracers_count"
        ",current_icu,current_icu_total,icu_beds\n"
        "2020-08-17,36,10,10,90,1,,,\n"
        "2020-08-18,36,20,20,180,2,,,\n"
        "2020-08-19,36,,,,3,,,\n"
        "2020-08-20,36,40,40,360,4,,,\n"
    )
    timeseries = _fips_csv_to_one_region(data, Region.from_fips("36"))

    data = io.StringIO(
        "date,fips,Rt_indicator,Rt_indicator_ci90,intervention,all_hospitalized,beds,infected_c\n"
        "2020-08-17,36,1.1,.1,0,1,10,\n"
        "2020-08-18,36,1.2,.1,0,1,10,\n"
        "2020-08-19,36,1.1,.2,0,1,10,\n"
        "2020-08-20,36,1.1,.1,0,1,10,\n"
        "2020-09-21,36,,,0,1,10,\n"
    )
    data = common_df.read_csv(data, set_index=False)
    model_output = CANPyseirLocationOutput(data)

    latest = {
        CommonFields.POPULATION: 100_000,
        CommonFields.FIPS: "36",
        CommonFields.STATE: "NY",
        CommonFields.ICU_TYPICAL_OCCUPANCY_RATE: 0.5,
        CommonFields.ICU_BEDS: 25,
    }
    results, _ = top_level_metrics.calculate_metrics_for_timeseries(
        timeseries, latest, model_output
    )
    expected = _build_metrics_df(
        "2020-08-17,36,,,,1.1,.1\n"
        "2020-08-18,36,10,0.1,0.04,1.2,.1\n"
        "2020-08-19,36,,0.1,,1.1,.2\n"
        "2020-08-20,36,,0.1,,1.1,.1\n"
    )
    pd.testing.assert_frame_equal(expected, results)


def test_calculate_contact_tracers():

    cases = _series_with_date_index([0.0, 1.0, 4.0])
    contact_tracers = _series_with_date_index([5, 5, 5])

    results = top_level_metrics.calculate_contact_tracers(cases, contact_tracers)
    expected = _series_with_date_index([np.nan, 1.0, 0.5])
    pd.testing.assert_series_equal(results, expected)


def test_calculate_contact_tracers_no_tracers():

    cases = _series_with_date_index([0.0, 1.0, 4.0])
    contact_tracers = _series_with_date_index([np.nan, np.nan, np.nan])

    results = top_level_metrics.calculate_contact_tracers(cases, contact_tracers)
    expected = _series_with_date_index([np.nan, np.nan, np.nan])
    pd.testing.assert_series_equal(results, expected)


def test_calculate_latest_rt():
    prev_rt = 1.0
    prev_rt_ci90 = 0.2
    data = _build_metrics_df(
        f"2020-08-13,36,10,0.1,0.06,{prev_rt},{prev_rt_ci90}\n"
        "2020-08-20,36,10,0.1,0.08,2.0,0.2\n"
    )
    metrics = top_level_metrics.calculate_latest_metrics(data, None)
    assert metrics.infectionRate == prev_rt
    assert metrics.infectionRateCI90 == prev_rt_ci90


def test_lookback_days():
    prev_rt = 1.0
    prev_rt_ci90 = 0.2
    data = _build_metrics_df(
        f"2020-08-12,36,10,0.1,0.06,{prev_rt},{prev_rt_ci90}\n"
        f"2020-08-13,36,10,0.1,0.06,,\n"
        "2020-08-20,,,,,,\n"
    )
    metrics = top_level_metrics.calculate_latest_metrics(data, None)
    assert not metrics.caseDensity
    assert not metrics.testPositivityRatio
    assert not metrics.infectionRate
    assert not metrics.infectionRateCI90


def test_calculate_latest_rt_no_previous_row():
    data = _build_metrics_df("2020-08-20,36,10,0.1,0.08,2.0,0.2\n")
    metrics = top_level_metrics.calculate_latest_metrics(data, None)
    assert not metrics.infectionRate
    assert not metrics.infectionRateCI90


def test_calculate_latest_rt_no_rt():
    data = _build_metrics_df("2020-08-20,36,10,0.1,0.08,,\n")
    metrics = top_level_metrics.calculate_latest_metrics(data, None)
    assert not metrics.infectionRate
    assert not metrics.infectionRateCI90


def test_calculate_latest_different_latest_days():
    prev_rt = 1.0
    prev_rt_ci90 = 0.2
    data = _build_metrics_df(
        f"2020-08-13,36,10,0.1,0.06,{prev_rt},{prev_rt_ci90}\n"
        "2020-08-20,36,,0.20,0.08,2.01,0.2\n"
    )
    expected_metrics = can_api_definition.Metrics(
        testPositivityRatio=0.2,
        caseDensity=10,
        contactTracerCapacityRatio=0.08,
        infectionRate=prev_rt,
        infectionRateCI90=prev_rt_ci90,
        icuHeadroomRatio=None,
        icuHeadroomDetails=None,
    )
    metrics = top_level_metrics.calculate_latest_metrics(data, None, max_lookback_days=8)
    assert metrics == expected_metrics
