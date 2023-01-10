from datetime import datetime
from typing import List, Dict, Optional
import pandas as pd
from math import isinf

from api.can_api_v2_definition import (
    Actuals,
    ActualsTimeseriesRow,
    AggregateRegionSummaryWithTimeseries,
    Annotations,
    CdcTransmissionLevelTimeseriesRow,
    CommunityLevelsTimeseriesRow,
    FieldAnnotations,
    AggregateFlattenedTimeseries,
    DemographicDistributions,
    Metrics,
    RiskLevels,
    CDCTransmissionLevel,
    RegionSummary,
    RegionSummaryWithTimeseries,
    RegionTimeseriesRowWithHeader,
    MetricsTimeseriesRow,
    RiskLevelTimeseriesRow,
)
from datapublic.common_fields import CommonFields

from api.can_api_v2_definition import AnomalyAnnotation
from api.can_api_v2_definition import FieldSource
from api.can_api_v2_definition import FieldSourceType

from libs.datasets import timeseries
from libs.datasets.tail_filter import TagField
from libs.datasets.timeseries import OneRegionTimeseriesDataset

from libs.metrics.community_levels import CommunityLevel


METRIC_SOURCES_NOT_FOUND_MESSAGE = "Unable to find provenance in FieldSourceType enum"
METRIC_MULTIPLE_SOURCE_URLS_MESSAGE = "More than one source_url for a field"
METRIC_MULTIPLE_SOURCE_TYPES_MESSAGE = "More than one provenance for a field"

USA_VACCINATION_START_DATE = datetime(2020, 12, 14)
USA_BIVALENT_VACCINATION_START_DATE = datetime(2022, 9, 1)


def _build_distributions(
    distributions: Dict[str, Dict[str, int]]
) -> Optional[DemographicDistributions]:
    data = {
        "age": distributions.get("age"),
        "race": distributions.get("race"),
        "ethnicity": distributions.get("ethnicity"),
        "sex": distributions.get("sex"),
    }

    # If there is no demographic data, do not create a DemographicDistributions
    # object, simply return none
    if not any(value for value in data.values()):
        return None

    return DemographicDistributions(**data)


def _build_actuals(actual_data: dict, distributions_by_field: Optional[Dict] = None) -> Actuals:
    """Generate actuals entry.

    Args:
        actual_data: Dictionary of data, generally derived one of the combined datasets.
        intervention: Current state level intervention.
        distributions_by_field: Optional dictionary of demographic distributions for API fields.
    """
    distributions_by_field = distributions_by_field or {}
    vaccines_administered_demographics = _build_distributions(
        distributions_by_field.get(CommonFields.VACCINES_ADMINISTERED, {})
    )
    vaccines_initiated_demographics = _build_distributions(
        distributions_by_field.get(CommonFields.VACCINATIONS_INITIATED, {})
    )

    # HACK: Unbreak Nebraska counties.
    vaccinations_initiated = actual_data.get(CommonFields.VACCINATIONS_INITIATED)
    if vaccinations_initiated and isinf(vaccinations_initiated):
        vaccinations_initiated = None

    return Actuals(
        cases=actual_data.get(CommonFields.CASES),
        deaths=actual_data.get(CommonFields.DEATHS),
        positiveTests=actual_data.get(CommonFields.POSITIVE_TESTS),
        negativeTests=actual_data.get(CommonFields.NEGATIVE_TESTS),
        contactTracers=actual_data.get(CommonFields.CONTACT_TRACERS_COUNT),
        hospitalBeds={
            "capacity": actual_data.get(CommonFields.STAFFED_BEDS),
            "currentUsageCovid": actual_data.get(CommonFields.CURRENT_HOSPITALIZED),
            "currentUsageTotal": actual_data.get(CommonFields.HOSPITAL_BEDS_IN_USE_ANY),
            "weeklyCovidAdmissions": actual_data.get(
                CommonFields.WEEKLY_NEW_HOSPITAL_ADMISSIONS_COVID
            ),
        },
        hsaHospitalBeds={
            "capacity": actual_data.get(CommonFields.STAFFED_BEDS_HSA),
            "currentUsageCovid": actual_data.get(CommonFields.CURRENT_HOSPITALIZED_HSA),
            "currentUsageTotal": actual_data.get(CommonFields.HOSPITAL_BEDS_IN_USE_ANY_HSA),
            "weeklyCovidAdmissions": actual_data.get(
                CommonFields.WEEKLY_NEW_HOSPITAL_ADMISSIONS_COVID_HSA
            ),
        },
        icuBeds={
            "capacity": actual_data.get(CommonFields.ICU_BEDS),
            "currentUsageCovid": actual_data.get(CommonFields.CURRENT_ICU),
            "currentUsageTotal": actual_data.get(CommonFields.CURRENT_ICU_TOTAL),
        },
        hsaIcuBeds={
            "capacity": actual_data.get(CommonFields.ICU_BEDS_HSA),
            "currentUsageCovid": actual_data.get(CommonFields.CURRENT_ICU_HSA),
            "currentUsageTotal": actual_data.get(CommonFields.CURRENT_ICU_TOTAL_HSA),
        },
        newCases=actual_data.get(CommonFields.NEW_CASES),
        newDeaths=actual_data.get(CommonFields.NEW_DEATHS),
        vaccinesDistributed=actual_data.get(CommonFields.VACCINES_DISTRIBUTED),
        vaccinationsInitiated=vaccinations_initiated,
        vaccinationsCompleted=actual_data.get(CommonFields.VACCINATIONS_COMPLETED),
        vaccinationsAdditionalDose=actual_data.get(CommonFields.VACCINATIONS_ADDITIONAL_DOSE),
        vaccinationsFall2022BivalentBooster=actual_data.get(
            CommonFields.VACCINATIONS_2022_FALL_BIVALENT_DOSE
        ),
        vaccinesAdministered=actual_data.get(CommonFields.VACCINES_ADMINISTERED),
        vaccinesAdministeredDemographics=vaccines_administered_demographics,
        vaccinationsInitiatedDemographics=vaccines_initiated_demographics,
    )


def build_region_summary(
    one_region: timeseries.OneRegionTimeseriesDataset,
    latest_metrics: Optional[Metrics],
    risk_levels: RiskLevels,
    cdc_transmission_level: CDCTransmissionLevel,
    community_levels: CommunityLevel,
    log,
) -> RegionSummary:
    latest_values = one_region.latest

    region = one_region.region
    distributions = one_region.demographic_distributions_by_field
    actuals = _build_actuals(latest_values, distributions_by_field=distributions)
    # HACK: HSA codes are converted to numerics somewhere in the pipeline, this
    # transforms them back into 3 character strings.
    hsa = latest_values.get(CommonFields.HSA)
    hsa = str(int(hsa)).zfill(3) if hsa is not None else None
    return RegionSummary(
        fips=region.fips_for_api,
        country=region.country,
        state=region.state,
        county=latest_values.get(CommonFields.COUNTY),
        level=region.level,
        lat=latest_values.get(CommonFields.LATITUDE),
        long=latest_values.get(CommonFields.LONGITUDE),
        population=latest_values[CommonFields.POPULATION],
        hsa=hsa,
        hsaName=latest_values.get(CommonFields.HSA_NAME),
        hsaPopulation=latest_values.get(CommonFields.HSA_POPULATION),
        actuals=actuals,
        metrics=latest_metrics,
        riskLevels=risk_levels,
        cdcTransmissionLevel=cdc_transmission_level,
        communityLevels=community_levels,
        lastUpdatedDate=datetime.utcnow(),
        locationId=region.location_id,
        url=latest_values.get(CommonFields.CAN_LOCATION_PAGE_URL),
        annotations=build_annotations(one_region, log),
    )


ACTUALS_NAME_TO_COMMON_FIELD = {
    "cases": CommonFields.CASES,
    "deaths": CommonFields.DEATHS,
    "positiveTests": CommonFields.POSITIVE_TESTS,
    "negativeTests": CommonFields.NEGATIVE_TESTS,
    "contactTracers": CommonFields.CONTACT_TRACERS_COUNT,
    "hospitalBeds": CommonFields.HOSPITAL_BEDS_IN_USE_ANY,
    "hsaHospitalBeds": CommonFields.HOSPITAL_BEDS_IN_USE_ANY_HSA,
    "icuBeds": CommonFields.ICU_BEDS,
    "hsaIcuBeds": CommonFields.ICU_BEDS_HSA,
    "newCases": CommonFields.NEW_CASES,
    "newDeaths": CommonFields.NEW_DEATHS,
    "vaccinesAdministered": CommonFields.VACCINES_ADMINISTERED,
    "vaccinesDistributed": CommonFields.VACCINES_DISTRIBUTED,
    "vaccinationsInitiated": CommonFields.VACCINATIONS_INITIATED,
    "vaccinationsCompleted": CommonFields.VACCINATIONS_COMPLETED,
    "vaccinationsAdditionalDose": CommonFields.VACCINATIONS_ADDITIONAL_DOSE,
    "vaccinationsFall2022BivalentBooster": CommonFields.VACCINATIONS_2022_FALL_BIVALENT_DOSE,
}


# Mapping used to connect each metric to a data source, in order to build the Annotations.
METRICS_NAME_TO_COMMON_FIELD = {
    "contactTracerCapacityRatio": CommonFields.CONTACT_TRACERS_COUNT,
    "caseDensity": CommonFields.CASES,
    "weeklyNewCasesPer100k": CommonFields.CASES,
    "infectionRate": CommonFields.CASES,
    "testPositivityRatio": CommonFields.TEST_POSITIVITY,
    "infectionRateCI90": CommonFields.CASES,
    "vaccinationsInitiatedRatio": CommonFields.VACCINATIONS_INITIATED_PCT,
    "vaccinationsCompletedRatio": CommonFields.VACCINATIONS_COMPLETED_PCT,
    "vaccinationsAdditionalDoseRatio": CommonFields.VACCINATIONS_ADDITIONAL_DOSE_PCT,
    "vaccinationsFall2022BivalentBoosterRatio": CommonFields.VACCINATIONS_BIVALENT_DOSE_PCT,
    "icuCapacityRatio": CommonFields.CURRENT_ICU,
    "bedsWithCovidPatientsRatio": CommonFields.CURRENT_HOSPITALIZED,
    "weeklyCovidAdmissionsPer100k": CommonFields.WEEKLY_NEW_HOSPITAL_ADMISSIONS_COVID,
}


def build_annotations(one_region: OneRegionTimeseriesDataset, log) -> Annotations:
    assert one_region.tag_all_bucket.index.names == [TagField.VARIABLE, TagField.TYPE]
    name_and_common_field = [
        *ACTUALS_NAME_TO_COMMON_FIELD.items(),
        *METRICS_NAME_TO_COMMON_FIELD.items(),
    ]
    annotations = {
        annotations_name: _build_metric_annotations(one_region, field_name, log)
        for annotations_name, field_name in name_and_common_field
    }
    return Annotations(**annotations)


def _build_metric_annotations(
    tag_series: timeseries.OneRegionTimeseriesDataset, field_name: CommonFields, log
) -> Optional[FieldAnnotations]:

    sources = [
        FieldSource(type=_lookup_source_type(tag.type, field_name, log), url=tag.url, name=tag.name)
        for tag in tag_series.sources_all_bucket(field_name)
    ]

    if not sources:
        # Fall back to using provenance and source_url.
        # TODO(tom): Remove this block of code when we're pretty sure `source` has all the data
        #  we need.
        sources = _sources_from_provenance_and_source_url(field_name, tag_series, log)

    anomalies = tag_series.annotations_all_bucket(field_name)
    anomalies = [
        AnomalyAnnotation(
            date=tag.date, original_observation=tag.original_observation, type=tag.tag_type
        )
        for tag in anomalies
    ]

    if not sources and not anomalies:
        return None

    return FieldAnnotations(sources=sources, anomalies=anomalies)


def _sources_from_provenance_and_source_url(
    field_name: CommonFields, tag_series: timeseries.OneRegionTimeseriesDataset, log
) -> List[FieldSource]:
    sources_enum = set()
    for source_str in tag_series.provenance.get(field_name, []):
        sources_enum.add(_lookup_source_type(source_str, field_name, log))
    if not sources_enum:
        source_enum = None
    else:
        if len(sources_enum) > 1:
            log.warning(METRIC_MULTIPLE_SOURCE_TYPES_MESSAGE, field_name=field_name)
        source_enum = sources_enum.pop()

    source_urls = set(tag_series.source_url.get(field_name, []))
    if not source_urls:
        source_url = None
    else:
        if len(source_urls) > 1:
            log.warning(
                METRIC_MULTIPLE_SOURCE_URLS_MESSAGE,
                field_name=field_name,
                urls=list(sorted(source_urls)),
            )
        # If multiple URLs actually happens in a meaningful way consider doing something better than
        # returning one at random.
        source_url = source_urls.pop()

    if source_url or source_enum:
        return [FieldSource(type=source_enum, url=source_url)]
    else:
        return []


def _lookup_source_type(source_str, field_name, log) -> FieldSourceType:
    source_enum = FieldSourceType.get(source_str)
    if source_enum is None:
        source_enum = FieldSourceType.OTHER
        log.info(
            METRIC_SOURCES_NOT_FOUND_MESSAGE, field_name=field_name, provenance=source_str,
        )
    return source_enum


def build_region_timeseries(
    region_summary: RegionSummary,
    timeseries: OneRegionTimeseriesDataset,
    metrics_timeseries: pd.DataFrame,
    risk_level_timeseries: pd.DataFrame,
    cdc_transmission_level_timeseries: pd.DataFrame,
    community_levels_timeseries: pd.DataFrame,
) -> RegionSummaryWithTimeseries:
    actuals_timeseries = []

    for row in timeseries.yield_records():
        # Timeseries records don't have population
        row[CommonFields.POPULATION] = region_summary.population
        actual = _build_actuals(row).dict()

        # Don't include vaccinations in timeseries before first possible vaccination
        # date to not bloat timeseries.
        if row[CommonFields.DATE] < USA_VACCINATION_START_DATE:
            del actual["vaccinesAdministered"]
            del actual["vaccinesDistributed"]
            del actual["vaccinationsInitiated"]
            del actual["vaccinationsCompleted"]
            del actual["vaccinationsAdditionalDose"]

        if row[CommonFields.DATE] < USA_BIVALENT_VACCINATION_START_DATE:
            del actual["vaccinationsFall2022BivalentBooster"]

        timeseries_row = ActualsTimeseriesRow(**actual, date=row[CommonFields.DATE])
        actuals_timeseries.append(timeseries_row)

    metrics_rows = []
    for metric_row in metrics_timeseries.to_dict(orient="records"):
        # Don't include vaccinations in timeseries before first possible vaccination
        # date to not bloat timeseries.
        if metric_row[CommonFields.DATE] < USA_VACCINATION_START_DATE:
            del metric_row["vaccinationsInitiatedRatio"]
            del metric_row["vaccinationsCompletedRatio"]
            del metric_row["vaccinationsAdditionalDoseRatio"]
        if metric_row[CommonFields.DATE] < USA_BIVALENT_VACCINATION_START_DATE:
            del metric_row["vaccinationsFall2022BivalentBoosterRatio"]
        metrics_rows.append(MetricsTimeseriesRow(**metric_row))

    risk_level_rows = [
        RiskLevelTimeseriesRow(**row) for row in risk_level_timeseries.to_dict(orient="records")
    ]
    cdc_transmission_level_rows = [
        CdcTransmissionLevelTimeseriesRow(**row)
        for row in cdc_transmission_level_timeseries.to_dict(orient="records")
    ]
    community_levels_timeseries = community_levels_timeseries.where(
        pd.notna(community_levels_timeseries), None
    )
    community_levels_rows = [
        CommunityLevelsTimeseriesRow(**row)
        for row in community_levels_timeseries.to_dict(orient="records")
    ]
    region_summary_data = {key: getattr(region_summary, key) for (key, _) in region_summary}
    return RegionSummaryWithTimeseries(
        **region_summary_data,
        actualsTimeseries=actuals_timeseries,
        metricsTimeseries=metrics_rows,
        riskLevelsTimeseries=risk_level_rows,
        cdcTransmissionLevelTimeseries=cdc_transmission_level_rows,
        communityLevelsTimeseries=community_levels_rows
    )


def build_bulk_flattened_timeseries(
    bulk_timeseries: AggregateRegionSummaryWithTimeseries,
) -> AggregateFlattenedTimeseries:
    rows = []
    for region_timeseries in bulk_timeseries.__root__:
        # Iterate through each state or county in data, adding summary data to each
        # timeseries row.
        summary_data = {
            "country": region_timeseries.country,
            "county": region_timeseries.county,
            "state": region_timeseries.state,
            "fips": region_timeseries.fips,
            "lat": region_timeseries.lat,
            "long": region_timeseries.long,
            "locationId": region_timeseries.locationId,
            "lastUpdatedDate": datetime.utcnow(),
            "hsa": region_timeseries.hsa,
            "hsaName": region_timeseries.hsaName,
            "hsaPopulation": region_timeseries.hsaPopulation,
        }
        actuals_by_date = {row.date: row for row in region_timeseries.actualsTimeseries}
        metrics_by_date = {row.date: row for row in region_timeseries.metricsTimeseries}
        risk_levels_by_date = {row.date: row for row in region_timeseries.riskLevelsTimeseries}
        cdc_transmission_levels_by_date = {
            row.date: row for row in region_timeseries.cdcTransmissionLevelTimeseries
        }
        community_levels_by_date = {
            row.date: row for row in region_timeseries.communityLevelsTimeseries
        }
        dates = sorted({*metrics_by_date.keys(), *actuals_by_date.keys()})
        for date in dates:
            data = {
                "date": date,
                "actuals": actuals_by_date.get(date),
                "metrics": metrics_by_date.get(date),
                "riskLevels": risk_levels_by_date.get(date),
                "cdcTransmissionLevel": cdc_transmission_levels_by_date.get(
                    date
                ).cdcTransmissionLevel,
                "communityLevels": community_levels_by_date.get(date),
            }
            data.update(summary_data)
            row = RegionTimeseriesRowWithHeader(**data)
            rows.append(row)

    return AggregateFlattenedTimeseries(__root__=rows)
