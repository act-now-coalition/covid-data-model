import datetime

import pytest

from api.can_api_v2_definition import Actuals
from api.can_api_v2_definition import RegionSummary
from libs.datasets import can_model_output_schema as schema
from libs.datasets import combined_datasets
from libs.datasets.sources.can_pyseir_location_output import CANPyseirLocationOutput
from libs.functions import build_api_v2
from libs.pipelines import api_v2_pipeline


@pytest.mark.parametrize(
    "include_model_output,rt_null", [(True, True), (True, False), (False, False)]
)
def test_build_summary_for_fips(
    include_model_output, rt_null, nyc_model_output_path, nyc_fips, nyc_region
):
    us_latest = combined_datasets.load_us_latest_dataset()
    us_timeseries = combined_datasets.load_us_timeseries_dataset()
    nyc_latest = us_latest.get_record_for_fips(nyc_fips)
    model_output = None
    expected_projections = None

    if include_model_output:
        model_output = CANPyseirLocationOutput.load_from_path(nyc_model_output_path)

        if rt_null:
            model_output.data[schema.RT_INDICATOR] = None
            model_output.data[schema.RT_INDICATOR_CI90] = None
        rt = model_output.latest_rt
        rt_ci_90 = model_output.latest_rt_ci90

    fips_timeseries = us_timeseries.get_one_region(nyc_region)

    metrics_series, latest_metric = api_v2_pipeline.generate_metrics_and_latest(
        fips_timeseries, model_output
    )
    assert latest_metric
    summary = build_api_v2.build_region_summary(nyc_latest, latest_metric)
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


@pytest.mark.parametrize("include_projections", [True])
def test_generate_timeseries_for_fips(
    include_projections, nyc_model_output_path, nyc_fips, nyc_region
):
    us_latest = combined_datasets.load_us_latest_dataset()
    us_timeseries = combined_datasets.load_us_timeseries_dataset()

    nyc_latest = us_latest.get_record_for_fips(nyc_fips)
    nyc_timeseries = us_timeseries.get_one_region(nyc_region)
    model_output = CANPyseirLocationOutput.load_from_path(nyc_model_output_path)
    metrics_series, latest_metric = api_v2_pipeline.generate_metrics_and_latest(
        nyc_timeseries, model_output
    )

    region_summary = build_api_v2.build_region_summary(nyc_latest, latest_metric)
    region_timeseries = build_api_v2.build_region_timeseries(
        region_summary, nyc_timeseries, metrics_series
    )

    summary = build_api_v2.build_region_summary(nyc_latest, latest_metric)

    assert summary.dict() == region_timeseries.region_summary.dict()
    # Double checking that serialized json does not contain NaNs, all values should
    # be serialized using the simplejson wrapper.
    assert "NaN" not in region_timeseries.json()
