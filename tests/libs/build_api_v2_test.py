import datetime

import pytest
import structlog
from datapublic.common_fields import DemographicBucket
from datapublic.common_fields import CommonFields

from api.can_api_v2_definition import Actuals
from api.can_api_v2_definition import FieldAnnotations
from api.can_api_v2_definition import FieldSource
from api.can_api_v2_definition import FieldSourceType
from api.can_api_v2_definition import RegionSummary
from api.can_api_v2_definition import RiskLevels
from api.can_api_v2_definition import CDCTransmissionLevel
from api.can_api_v2_definition import DemographicDistributions
from api.can_api_v2_definition import Metrics
from libs.metrics import top_level_metric_risk_levels
from libs.metrics import cdc_transmission_levels
from libs import build_api_v2
from libs.pipelines import api_v2_pipeline
from tests import test_helpers
from tests.test_helpers import TimeseriesLiteral
from libs.pipeline import Region


@pytest.mark.parametrize(
    "include_model_output,rt_null", [(True, True), (True, False), (False, False)]
)
def test_build_summary_for_fips(
    include_model_output: bool, rt_null: bool, nyc_region, nyc_rt_dataset
):
    us_timeseries = test_helpers.load_test_dataset()

    if include_model_output:
        if rt_null:
            nyc_rt_dataset = None
    else:
        nyc_rt_dataset = None

    fips_timeseries = us_timeseries.get_one_region(nyc_region)
    nyc_latest = fips_timeseries.latest
    log = structlog.get_logger()
    field_source_usafacts = FieldSource(
        name="USAFacts",
        type=FieldSourceType.USA_FACTS,
        url="https://usafacts.org/issues/coronavirus/",
    )

    metrics_series, latest_metric = api_v2_pipeline.generate_metrics_and_latest(
        fips_timeseries, nyc_rt_dataset, log,
    )
    risk_levels = top_level_metric_risk_levels.calculate_risk_level_from_metrics(latest_metric)
    transmission_level = cdc_transmission_levels.calculate_transmission_level_from_metrics(
        latest_metric
    )
    assert latest_metric
    summary = build_api_v2.build_region_summary(
        fips_timeseries, latest_metric, risk_levels, transmission_level, log
    )
    field_source_hhshospital = FieldSource(
        name="Department of Health and Human Services",
        type=FieldSourceType.OTHER,
        url="https://healthdata.gov/Hospital/COVID-19-Reported-Patient-Impact-and-Hospital-Capa/anag-cw7u",
    )
    field_source_cdcvaccine = FieldSource(
        name="Centers for Disease Control and Prevention",
        type=FieldSourceType.OTHER,
        url="https://data.cdc.gov/Vaccinations/COVID-19-Vaccinations-in-the-United-States-County/8xkx-amqh",
    )
    expected = RegionSummary(
        population=nyc_latest["population"],
        state="NY",
        country="US",
        level="county",
        county="New York County",
        fips="36061",
        locationId="iso1:us#iso2:us-ny#fips:36061",
        lat=None,
        long=None,
        metrics=latest_metric,
        riskLevels=risk_levels,
        cdcTransmissionLevel=transmission_level,
        actuals=Actuals(
            cases=nyc_latest["cases"],
            deaths=nyc_latest["deaths"],
            positiveTests=nyc_latest["positive_tests"],
            negativeTests=nyc_latest["negative_tests"],
            hospitalBeds={
                "capacity": nyc_latest["staffed_beds"],
                "currentUsageCovid": nyc_latest["current_hospitalized"],
                "currentUsageTotal": nyc_latest["hospital_beds_in_use_any"],
            },
            icuBeds={
                "capacity": nyc_latest["icu_beds"],
                "totalCapacity": nyc_latest["icu_beds"],
                "currentUsageCovid": nyc_latest["current_icu"],
                "currentUsageTotal": nyc_latest["current_icu_total"],
            },
            contactTracers=nyc_latest["contact_tracers_count"],
            newCases=nyc_latest["new_cases"],
            newDeaths=nyc_latest["new_deaths"],
            vaccinesDistributed=nyc_latest["vaccines_distributed"],
            vaccinesAdministered=nyc_latest["vaccines_administered"],
            vaccinationsInitiated=nyc_latest["vaccinations_initiated"],
            vaccinationsCompleted=nyc_latest.get("vaccinations_completed"),
            vaccinesAdministeredDemographics=None,
            vaccinesInitiatedDemographics=None,
        ),
        annotations=dict(
            cases=FieldAnnotations(sources=[field_source_usafacts], anomalies=[],),
            deaths=FieldAnnotations(sources=[field_source_usafacts], anomalies=[],),
            hospitalBeds=FieldAnnotations(sources=[field_source_hhshospital], anomalies=[]),
            icuBeds=FieldAnnotations(sources=[field_source_hhshospital], anomalies=[]),
            newDeaths=FieldAnnotations(
                anomalies=[
                    {
                        "date": datetime.date(2021, 3, 13),
                        "original_observation": 110.0,
                        "type": "zscore_outlier",
                    },
                    {
                        "date": datetime.date(2021, 4, 9),
                        "original_observation": 182.0,
                        "type": "zscore_outlier",
                    },
                ],
                sources=[],
            ),
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
            vaccinationsCompleted=FieldAnnotations(
                sources=[field_source_cdcvaccine], anomalies=[],
            ),
            vaccinationsInitiated=FieldAnnotations(
                sources=[field_source_cdcvaccine], anomalies=[],
            ),
            caseDensity=FieldAnnotations(sources=[field_source_usafacts], anomalies=[],),
            icuCapacityRatio=FieldAnnotations(sources=[field_source_hhshospital], anomalies=[]),
            infectionRate=FieldAnnotations(sources=[field_source_usafacts], anomalies=[],),
            infectionRateCI90=FieldAnnotations(sources=[field_source_usafacts], anomalies=[],),
        ),
        lastUpdatedDate=datetime.datetime.utcnow(),
        url="https://covidactnow.org/us/new_york-ny/county/bronx_county",
    )
    assert expected.dict() == summary.dict()


def test_generate_timeseries_for_fips(nyc_region, nyc_rt_dataset):
    us_timeseries = test_helpers.load_test_dataset()

    nyc_timeseries = us_timeseries.get_one_region(nyc_region)
    nyc_latest = nyc_timeseries.latest
    log = structlog.get_logger()
    metrics_series, latest_metric = api_v2_pipeline.generate_metrics_and_latest(
        nyc_timeseries, nyc_rt_dataset, log
    )
    risk_levels = top_level_metric_risk_levels.calculate_risk_level_from_metrics(latest_metric)
    risk_timeseries = top_level_metric_risk_levels.calculate_risk_level_timeseries(metrics_series)
    transmission_level = cdc_transmission_levels.calculate_transmission_level_from_metrics(
        latest_metric
    )
    transmission_level_timeseries = cdc_transmission_levels.calculate_transmission_level_timeseries(
        metrics_series
    )

    region_summary = build_api_v2.build_region_summary(
        nyc_timeseries, latest_metric, risk_levels, transmission_level, log
    )
    region_timeseries = build_api_v2.build_region_timeseries(
        region_summary,
        nyc_timeseries,
        metrics_series,
        risk_timeseries,
        transmission_level_timeseries,
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

    summary = build_api_v2.build_region_summary(
        nyc_timeseries, latest_metric, risk_levels, transmission_level, log
    )

    assert summary.dict() == region_timeseries.region_summary.dict()
    # Double checking that serialized json does not contain NaNs, all values should
    # be serialized using the simplejson wrapper.
    assert "NaN" not in region_timeseries.json()


def test_multiple_distributions():
    """All time-series within a variable are treated as a unit when combining"""
    all_bucket = DemographicBucket("all")
    age_20s = DemographicBucket("age:20-29")
    age_30s = DemographicBucket("age:30-39")
    region_ak = Region.from_state("AK")
    region_ca = Region.from_state("CA")
    log = structlog.get_logger()

    ds2 = test_helpers.build_dataset(
        {
            region_ca: {
                CommonFields.VACCINES_ADMINISTERED: {
                    all_bucket: TimeseriesLiteral([3, 4], provenance="ds2_ca_m1_30s"),
                    age_30s: TimeseriesLiteral([3, 4], provenance="ds2_ca_m1_30s"),
                },
                CommonFields.DEATHS: {
                    age_30s: TimeseriesLiteral([6, 7], provenance="ds2_ca_m2_30s")
                },
            },
        },
        static_by_region_then_field_name={region_ca: {CommonFields.POPULATION: 10000}},
    )

    one_region = ds2.get_one_region(region_ca)
    summary = build_api_v2.build_region_summary(
        one_region, Metrics.empty(), RiskLevels.empty(), CDCTransmissionLevel.UNKNOWN, log
    )
    expected_demographics = DemographicDistributions(age={"30-39": 4})
    assert summary.actuals.vaccinesAdministeredDemographics == expected_demographics
    assert summary.actuals.vaccinationsInitiatedDemographics is None
