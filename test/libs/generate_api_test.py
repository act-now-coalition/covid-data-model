import datetime
import pathlib
import tempfile

import pytest

from api.can_api_definition import (
    Actuals,
    Metrics,
    Projections,
    RegionSummary,
    RegionSummaryWithTimeseries,
    ResourceUsageProjection,
)
from libs.datasets import can_model_output_schema as schema
from libs.datasets import combined_datasets
from libs.datasets.sources.can_pyseir_location_output import CANPyseirLocationOutput
from libs.enums import Intervention
from libs.functions import generate_api
from libs.pipelines import api_pipeline


@pytest.mark.parametrize(
    "include_projections,rt_null", [(True, True), (True, False), (False, False)]
)
def test_build_summary_for_fips(include_projections, rt_null, nyc_model_output_path, nyc_fips):

    us_latest = combined_datasets.load_us_latest_dataset()
    us_timeseries = combined_datasets.load_us_timeseries_dataset()
    nyc_latest = us_latest.get_record_for_fips(nyc_fips)
    model_output = None
    expected_projections = None

    intervention = Intervention.OBSERVED_INTERVENTION
    if include_projections:
        model_output = CANPyseirLocationOutput.load_from_path(nyc_model_output_path)

        if rt_null:
            model_output.data[schema.RT_INDICATOR] = None
            model_output.data[schema.RT_INDICATOR_CI90] = None
        rt = model_output.latest_rt
        rt_ci_90 = model_output.latest_rt_ci90
        expected_projections = Projections(
            totalHospitalBeds=ResourceUsageProjection(
                peakShortfall=0, peakDate=datetime.date(2020, 4, 15), shortageStartDate=None
            ),
            ICUBeds=None,
            Rt=rt,
            RtCI90=rt_ci_90,
        )
        intervention = Intervention.STRONG_INTERVENTION

    fips_timeseries = us_timeseries.get_subset(None, fips=nyc_fips)
    metrics_series, latest_metric = api_pipeline.generate_metrics_and_latest_for_fips(
        fips_timeseries, nyc_latest
    )
    summary = generate_api.generate_region_summary(nyc_latest, latest_metric, model_output)

    expected = RegionSummary(
        population=nyc_latest["population"],
        stateName="New York",
        countyName="New York County",
        fips="36061",
        lat=None,
        long=None,
        metrics=latest_metric,
        actuals=Actuals(
            population=nyc_latest["population"],
            intervention="STRONG_INTERVENTION",
            cumulativeConfirmedCases=nyc_latest["cases"],
            cumulativeDeaths=nyc_latest["deaths"],
            cumulativePositiveTests=nyc_latest["positive_tests"],
            cumulativeNegativeTests=nyc_latest["negative_tests"],
            hospitalBeds={
                # Manually calculated from capacity calculation in generate_api.py
                "capacity": 12763,
                "totalCapacity": nyc_latest["max_bed_count"],
                "currentUsageCovid": 0,
                "currentUsageTotal": None,
                "typicalUsageRate": nyc_latest["all_beds_occupancy_rate"],
            },
            ICUBeds={
                "capacity": nyc_latest["icu_beds"],
                "totalCapacity": nyc_latest["icu_beds"],
                "currentUsageCovid": 0,
                "currentUsageTotal": 0,
                "typicalUsageRate": nyc_latest["icu_occupancy_rate"],
            },
            contactTracers=nyc_latest["contact_tracers_count"],
        ),
        lastUpdatedDate=datetime.datetime.utcnow(),
        projections=expected_projections,
    )
    import pprint

    pprint.pprint(expected.actuals.ICUBeds.dict())
    pprint.pprint(summary.actuals.ICUBeds.dict())
    assert expected.dict() == summary.dict()


@pytest.mark.parametrize("include_projections", [True])
def test_generate_timeseries_for_fips(include_projections, nyc_model_output_path, nyc_fips):

    us_latest = combined_datasets.load_us_latest_dataset()
    us_timeseries = combined_datasets.load_us_timeseries_dataset()

    nyc_latest = us_latest.get_record_for_fips(nyc_fips)
    nyc_timeseries = us_timeseries.get_subset(None, fips=nyc_fips)
    intervention = Intervention.OBSERVED_INTERVENTION
    model_output = CANPyseirLocationOutput.load_from_path(nyc_model_output_path)
    metrics_series, latest_metric = api_pipeline.generate_metrics_and_latest_for_fips(
        nyc_timeseries, nyc_latest
    )

    region_summary = generate_api.generate_region_summary(nyc_latest, latest_metric, model_output)
    region_timeseries = generate_api.generate_region_timeseries(
        region_summary, nyc_timeseries, metrics_series, model_output
    )

    summary = generate_api.generate_region_summary(nyc_latest, latest_metric, model_output)

    assert summary.dict() == region_timeseries.region_summary.dict()
    # Double checking that serialized json does not contain NaNs, all values should
    # be serialized using the simplejson wrapper.
    assert "NaN" not in region_timeseries.json()
