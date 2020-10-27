import datetime

import pytest

from api.can_api_definition import (
    Actuals,
    Projections,
    RegionSummary,
    ResourceUsageProjection,
)
from libs.datasets import combined_datasets
from libs.enums import Intervention
from libs.functions import generate_api
from libs.pipelines import api_pipeline


def test_build_summary_for_fips(
    nyc_region, nyc_rt_dataset, nyc_icu_dataset,
):
    us_latest = combined_datasets.load_us_latest_dataset()
    us_timeseries = combined_datasets.load_us_timeseries_dataset()
    nyc_latest = us_latest.get_record_for_fips(nyc_region.fips)

    fips_timeseries = us_timeseries.get_one_region(nyc_region)
    metrics_series, latest_metric = api_pipeline.generate_metrics_and_latest(
        fips_timeseries, nyc_rt_dataset, nyc_icu_dataset
    )
    assert latest_metric
    summary = generate_api.generate_region_summary(nyc_latest, latest_metric)
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
                "currentUsageCovid": None,
                "currentUsageTotal": None,
                "typicalUsageRate": nyc_latest["all_beds_occupancy_rate"],
            },
            ICUBeds={
                "capacity": nyc_latest["icu_beds"],
                "totalCapacity": nyc_latest["icu_beds"],
                "currentUsageCovid": None,
                "currentUsageTotal": None,
                "typicalUsageRate": nyc_latest["icu_occupancy_rate"],
            },
            contactTracers=nyc_latest["contact_tracers_count"],
        ),
        lastUpdatedDate=datetime.datetime.utcnow(),
        projections=None,
    )
    assert expected.dict() == summary.dict()


def test_generate_timeseries_for_fips(
    nyc_region, nyc_rt_dataset, nyc_icu_dataset,
):
    us_latest = combined_datasets.load_us_latest_dataset()
    us_timeseries = combined_datasets.load_us_timeseries_dataset()

    nyc_latest = us_latest.get_record_for_fips(nyc_region.fips)
    nyc_timeseries = us_timeseries.get_one_region(nyc_region)
    intervention = Intervention.OBSERVED_INTERVENTION
    metrics_series, latest_metric = api_pipeline.generate_metrics_and_latest(
        nyc_timeseries, nyc_rt_dataset, nyc_icu_dataset
    )

    region_summary = generate_api.generate_region_summary(nyc_latest, latest_metric)
    region_timeseries = generate_api.generate_region_timeseries(
        region_summary, nyc_timeseries, metrics_series
    )

    summary = generate_api.generate_region_summary(nyc_latest, latest_metric)

    assert summary.dict() == region_timeseries.region_summary.dict()
    # Double checking that serialized json does not contain NaNs, all values should
    # be serialized using the simplejson wrapper.
    assert "NaN" not in region_timeseries.json()
