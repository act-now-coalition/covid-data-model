import datetime

import pytest
import structlog

from api.can_api_v2_definition import Actuals
from api.can_api_v2_definition import Annotations
from api.can_api_v2_definition import FieldAnnotations
from api.can_api_v2_definition import FieldSource
from api.can_api_v2_definition import FieldSourceType
from api.can_api_v2_definition import RegionSummary
from libs.metrics import top_level_metric_risk_levels
from libs.datasets import combined_datasets
from libs import build_api_v2
from libs.pipelines import api_v2_pipeline


@pytest.mark.parametrize(
    "include_model_output,rt_null", [(True, True), (True, False), (False, False)]
)
def test_build_summary_for_fips(
    include_model_output: bool, rt_null: bool, nyc_region, nyc_icu_dataset, nyc_rt_dataset
):
    us_timeseries = combined_datasets.load_us_timeseries_dataset()

    if include_model_output:
        if rt_null:
            nyc_rt_dataset = None
    else:
        nyc_icu_dataset = None
        nyc_rt_dataset = None

    fips_timeseries = us_timeseries.get_one_region(nyc_region)
    nyc_latest = fips_timeseries.latest
    log = structlog.get_logger()
    usafacts_url = "https://usafacts.org/issues/coronavirus/"

    metrics_series, latest_metric = api_v2_pipeline.generate_metrics_and_latest(
        fips_timeseries, nyc_rt_dataset, nyc_icu_dataset, log,
    )
    risk_levels = top_level_metric_risk_levels.calculate_risk_level_from_metrics(latest_metric)
    assert latest_metric
    summary = build_api_v2.build_region_summary(fips_timeseries, latest_metric, risk_levels, log)
    expected = RegionSummary(
        population=nyc_latest["population"],
        state="NY",
        country="USA",
        level="county",
        county="New York County",
        fips="36061",
        locationId="iso1:us#iso2:us-ny#fips:36061",
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
                "currentUsageCovid": nyc_latest["current_hospitalized"],
                "currentUsageTotal": nyc_latest["hospital_beds_in_use_any"],
                "typicalUsageRate": nyc_latest["all_beds_occupancy_rate"],
            },
            icuBeds={
                "capacity": nyc_latest["icu_beds"],
                "totalCapacity": nyc_latest["icu_beds"],
                "currentUsageCovid": nyc_latest["current_icu"],
                "currentUsageTotal": nyc_latest["current_icu_total"],
                "typicalUsageRate": nyc_latest["icu_occupancy_rate"],
            },
            contactTracers=nyc_latest["contact_tracers_count"],
            newCases=nyc_latest["new_cases"],
            vaccinesDistributed=nyc_latest["vaccines_distributed"],
            vaccinationsInitiated=nyc_latest["vaccinations_initiated"],
            vaccinationsCompleted=nyc_latest.get("vaccinations_completed"),
        ),
        annotations=Annotations(
            cases=FieldAnnotations(
                sources=[FieldSource(type=FieldSourceType.USA_FACTS, url=usafacts_url)],
                anomalies=[],
            ),
            deaths=FieldAnnotations(
                sources=[FieldSource(type=FieldSourceType.USA_FACTS, url=usafacts_url)],
                anomalies=[],
            ),
            positiveTests=None,
            negativeTests=None,
            hospitalBeds=FieldAnnotations(
                sources=[FieldSource(type=FieldSourceType.HHSHospital)], anomalies=[]
            ),
            icuBeds=FieldAnnotations(
                sources=[FieldSource(type=FieldSourceType.HHSHospital)], anomalies=[]
            ),
            contactTracers=None,
            newCases=FieldAnnotations(
                sources=[],
                anomalies=[
                    {
                        "date": datetime.date(2020, 3, 17),
                        "original_observation": 166.0,
                        "type": "zscore_outlier",
                    },
                    {
                        "date": datetime.date(2020, 4, 15),
                        "original_observation": 1737.0,
                        "type": "zscore_outlier",
                    },
                ],
            ),
        ),
        lastUpdatedDate=datetime.datetime.utcnow(),
        url="https://covidactnow.org/us/new_york-ny/county/bronx_county",
    )
    assert expected.dict() == summary.dict()


def test_generate_timeseries_for_fips(nyc_region, nyc_rt_dataset, nyc_icu_dataset):
    us_timeseries = combined_datasets.load_us_timeseries_dataset()

    nyc_timeseries = us_timeseries.get_one_region(nyc_region)
    nyc_latest = nyc_timeseries.latest
    log = structlog.get_logger()
    metrics_series, latest_metric = api_v2_pipeline.generate_metrics_and_latest(
        nyc_timeseries, nyc_rt_dataset, nyc_icu_dataset, log
    )
    risk_levels = top_level_metric_risk_levels.calculate_risk_level_from_metrics(latest_metric)
    risk_timeseries = top_level_metric_risk_levels.calculate_risk_level_timeseries(metrics_series)

    region_summary = build_api_v2.build_region_summary(
        nyc_timeseries, latest_metric, risk_levels, log
    )
    region_timeseries = build_api_v2.build_region_timeseries(
        region_summary, nyc_timeseries, metrics_series, risk_timeseries
    )

    # Test vaccination fields aren't in before start date
    for actuals_row in region_timeseries.actualsTimeseries:
        row_data = actuals_row.dict(exclude_unset=True)
        if actuals_row.date < build_api_v2.USA_VACCINATION_START_DATE.date():
            assert "vaccinationsInitiated" not in row_data
        else:
            assert "vaccinationsInitiated" in row_data

    # Test vaccination fields aren't in before start date
    for row in region_timeseries.metricsTimeseries:
        row_data = row.dict(exclude_unset=True)
        if row.date < build_api_v2.USA_VACCINATION_START_DATE.date():
            assert "vaccinationsInitiatedRatio" not in row_data
        else:
            assert "vaccinationsInitiatedRatio" in row_data

    summary = build_api_v2.build_region_summary(nyc_timeseries, latest_metric, risk_levels, log)

    assert summary.dict() == region_timeseries.region_summary.dict()
    # Double checking that serialized json does not contain NaNs, all values should
    # be serialized using the simplejson wrapper.
    assert "NaN" not in region_timeseries.json()
