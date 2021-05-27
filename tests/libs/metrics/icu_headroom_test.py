import io
import pandas as pd
from covidactnow.datapublic import common_df
from covidactnow.datapublic.common_fields import CommonFields
from libs.metrics import icu_headroom

ICUMetricData = icu_headroom.ICUMetricData


def test_icu_metric_data_with_all_timeseries_actuals():

    data = io.StringIO(
        "date,fips,current_icu,current_icu_total,icu_beds\n"
        "2020-08-10,36,10,25,50\n"
        "2020-08-11,35,20,40,50\n"
    )
    data = common_df.read_csv(data, set_index=False).set_index(CommonFields.DATE)
    estimated_icu = pd.Series([20, 30], index=data.index)

    icu_data = ICUMetricData(data, estimated_icu, {}, 0.0, require_recent_data=False)
    pd.testing.assert_series_equal(icu_data.actual_current_icu_covid, data.current_icu)
    pd.testing.assert_series_equal(icu_data.estimated_current_icu_covid, estimated_icu)
    pd.testing.assert_series_equal(icu_data.actual_current_icu_total, data.current_icu_total)
    pd.testing.assert_series_equal(icu_data.total_icu_beds, data.icu_beds)

    non_covid, source = icu_data.current_icu_non_covid_with_source
    pd.testing.assert_series_equal(non_covid, pd.Series([15, 20], index=data.index))
    assert source is icu_headroom.NonCovidPatientsMethod.ACTUAL

    covid, source = icu_data.current_icu_covid_with_source
    pd.testing.assert_series_equal(covid, data.current_icu)
    assert source is icu_headroom.CovidPatientsMethod.ACTUAL


def test_icu_metric_data_with_estimated_from_total_icu_actuals():
    latest = {}
    data = io.StringIO(
        "date,fips,current_icu,current_icu_total,icu_beds\n"
        "2020-08-10,36,,25,50\n"
        "2020-08-11,35,,40,50\n"
    )
    data = common_df.read_csv(data, set_index=False).set_index(CommonFields.DATE)
    estimated_icu = pd.Series([20, 30], index=data.index)

    icu_data = ICUMetricData(data, estimated_icu, latest, 0.0, require_recent_data=False)
    assert not icu_data.actual_current_icu_covid

    non_covid, source = icu_data.current_icu_non_covid_with_source
    pd.testing.assert_series_equal(non_covid, pd.Series([5, 10], index=data.index))
    assert source is icu_headroom.NonCovidPatientsMethod.ESTIMATED_FROM_TOTAL_ICU_ACTUAL

    covid, source = icu_data.current_icu_covid_with_source
    pd.testing.assert_series_equal(covid, estimated_icu)
    assert source is icu_headroom.CovidPatientsMethod.ESTIMATED


def test_icu_metric_data_with_estimated_from_decomp_and_total_beds_timeseries():
    latest = {CommonFields.ICU_TYPICAL_OCCUPANCY_RATE: 0.5}
    data = io.StringIO(
        "date,fips,current_icu,current_icu_total,icu_beds\n"
        "2020-08-10,36,,,50\n"
        "2020-08-11,35,,,50\n"
    )
    data = common_df.read_csv(data, set_index=False).set_index(CommonFields.DATE)
    estimated_icu = pd.Series([20, 30], index=data.index)

    icu_data = ICUMetricData(data, estimated_icu, latest, 0.0, require_recent_data=False)
    assert not icu_data.actual_current_icu_covid

    non_covid, source = icu_data.current_icu_non_covid_with_source
    pd.testing.assert_series_equal(non_covid, pd.Series([25.0, 25.0], index=data.index))
    assert source is icu_headroom.NonCovidPatientsMethod.ESTIMATED_FROM_TYPICAL_UTILIZATION

    covid, source = icu_data.current_icu_covid_with_source
    pd.testing.assert_series_equal(covid, estimated_icu)
    assert source is icu_headroom.CovidPatientsMethod.ESTIMATED
