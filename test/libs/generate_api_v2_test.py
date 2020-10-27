import datetime

import pytest

from api.can_api_v2_definition import Actuals
from api.can_api_v2_definition import RegionSummary
from libs import top_level_metric_risk_levels
from libs.datasets import combined_datasets
from libs.functions import build_api_v2
from libs.pipelines import api_v2_pipeline


@pytest.mark.parametrize(
    "include_model_output,rt_null", [(True, True), (True, False), (False, False)]
)
def test_build_summary_for_fips(
    include_model_output, rt_null, nyc_region, nyc_icu_dataset, nyc_rt_dataset
):
    us_latest = combined_datasets.load_us_latest_dataset()
    us_timeseries = combined_datasets.load_us_timeseries_dataset()
    nyc_latest = us_latest.get_record_for_fips(nyc_region.fips)
    expected_projections = None

    if include_model_output:
        if rt_null:
            nyc_rt_dataset = None
    else:
        nyc_icu_dataset = None
        nyc_rt_dataset = None

    fips_timeseries = us_timeseries.get_one_region(nyc_region)

    metrics_series, latest_metric = api_v2_pipeline.generate_metrics_and_latest(
        fips_timeseries, nyc_rt_dataset, nyc_icu_dataset
    )
    risk_levels = top_level_metric_risk_levels.calculate_risk_level_from_metrics(latest_metric)
    assert latest_metric
    summary = build_api_v2.build_region_summary(nyc_latest, latest_metric, risk_levels)
    expected = RegionSummary(
        population=nyc_latest["population"],
        state="NY",
        country="USA",
        level="county",
        county="New York County",
        fips="36061",
        lat=None,
        long=None,
        metrics=latest_metric,
        riskLevels=risk_levels,
        actuals=Actuals(
            cases=nyc_latest["cases"],
            deaths=nyc_latest["deaths"],
            positiveTests=nyc_latest["positive_tests"],
            negativeTests=nyc_latest["negative_tests"],
            hospitalBeds={
                "capacity": nyc_latest["max_bed_count"],
                "currentUsageCovid": None,
                "currentUsageTotal": None,
                "typicalUsageRate": nyc_latest["all_beds_occupancy_rate"],
            },
            icuBeds={
                "capacity": nyc_latest["icu_beds"],
                "totalCapacity": nyc_latest["icu_beds"],
                "currentUsageCovid": None,
                "currentUsageTotal": None,
                "typicalUsageRate": nyc_latest["icu_occupancy_rate"],
            },
            contactTracers=nyc_latest["contact_tracers_count"],
        ),
        lastUpdatedDate=datetime.datetime.utcnow(),
    )
    assert expected.dict() == summary.dict()


def test_generate_timeseries_for_fips(nyc_region, nyc_rt_dataset, nyc_icu_dataset):
    us_latest = combined_datasets.load_us_latest_dataset()
    us_timeseries = combined_datasets.load_us_timeseries_dataset()

    nyc_latest = us_latest.get_record_for_fips(nyc_region.fips)
    nyc_timeseries = us_timeseries.get_one_region(nyc_region)
    metrics_series, latest_metric = api_v2_pipeline.generate_metrics_and_latest(
        nyc_timeseries, nyc_rt_dataset, nyc_icu_dataset
    )
    risk_levels = top_level_metric_risk_levels.calculate_risk_level_from_metrics(latest_metric)

    region_summary = build_api_v2.build_region_summary(nyc_latest, latest_metric, risk_levels)
    region_timeseries = build_api_v2.build_region_timeseries(
        region_summary, nyc_timeseries, metrics_series
    )

    summary = build_api_v2.build_region_summary(nyc_latest, latest_metric, risk_levels)

    assert summary.dict() == region_timeseries.region_summary.dict()
    # Double checking that serialized json does not contain NaNs, all values should
    # be serialized using the simplejson wrapper.
    assert "NaN" not in region_timeseries.json()
