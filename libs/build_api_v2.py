from datetime import datetime
from typing import Optional
import pandas as pd
from api.can_api_v2_definition import (
    Actuals,
    ActualsTimeseriesRow,
    Annotations,
    FieldAnnotations,
    AggregateFlattenedTimeseries,
    AggregateRegionSummary,
    Metrics,
    RiskLevels,
    RiskLevelsRow,
    RegionSummary,
    RegionSummaryWithTimeseries,
    RegionTimeseriesRowWithHeader,
    MetricsTimeseriesRow,
    RiskLevelTimeseriesRow,
)
from covidactnow.datapublic.common_fields import CommonFields

from api.can_api_v2_definition import AnomalyAnnotation
from api.can_api_v2_definition import FieldSource
from libs.datasets import timeseries
from libs.datasets.tail_filter import TagField
from libs.datasets.timeseries import OneRegionTimeseriesDataset


METRIC_SOURCES_NOT_FOUND_MESSAGE = "Unable to find provenance in FieldSource enum"
METRIC_MULTIPLE_SOURCE_URLS_MESSAGE = "More than one source_url for a field"

USA_VACCINATION_START_DATE = datetime(2020, 12, 14)


def _build_actuals(actual_data: dict) -> Actuals:
    """Generate actuals entry.

    Args:
        actual_data: Dictionary of data, generally derived one of the combined datasets.
        intervention: Current state level intervention.

    """
    return Actuals(
        cases=actual_data[CommonFields.CASES],
        deaths=actual_data[CommonFields.DEATHS],
        positiveTests=actual_data.get(CommonFields.POSITIVE_TESTS),
        negativeTests=actual_data.get(CommonFields.NEGATIVE_TESTS),
        contactTracers=actual_data.get(CommonFields.CONTACT_TRACERS_COUNT),
        hospitalBeds={
            "capacity": actual_data.get(CommonFields.MAX_BED_COUNT),
            "currentUsageCovid": actual_data.get(CommonFields.CURRENT_HOSPITALIZED),
            "currentUsageTotal": actual_data.get(CommonFields.HOSPITAL_BEDS_IN_USE_ANY),
            "typicalUsageRate": actual_data.get(CommonFields.ALL_BED_TYPICAL_OCCUPANCY_RATE),
        },
        icuBeds={
            "capacity": actual_data.get(CommonFields.ICU_BEDS),
            "currentUsageCovid": actual_data.get(CommonFields.CURRENT_ICU),
            "currentUsageTotal": actual_data.get(CommonFields.CURRENT_ICU_TOTAL),
            "typicalUsageRate": actual_data.get(CommonFields.ICU_TYPICAL_OCCUPANCY_RATE),
        },
        newCases=actual_data[CommonFields.NEW_CASES],
        vaccinesDistributed=actual_data.get(CommonFields.VACCINES_DISTRIBUTED),
        vaccinationsInitiated=actual_data.get(CommonFields.VACCINATIONS_INITIATED),
        # Vaccinations completed currently optional as data is not yet flowing through.
        # This will allow us to include vaccines completed data as soon as its scraped.
        vaccinationsCompleted=actual_data.get(CommonFields.VACCINATIONS_COMPLETED),
    )


def build_region_summary(
    one_region: timeseries.OneRegionTimeseriesDataset,
    latest_metrics: Optional[Metrics],
    risk_levels: RiskLevels,
    log,
) -> RegionSummary:
    latest_values = one_region.latest
    region = one_region.region

    actuals = _build_actuals(latest_values)
    return RegionSummary(
        fips=region.fips,
        country=region.country,
        state=region.state,
        county=latest_values.get(CommonFields.COUNTY),
        level=region.level,
        lat=latest_values.get(CommonFields.LATITUDE),
        long=latest_values.get(CommonFields.LONGITUDE),
        population=latest_values[CommonFields.POPULATION],
        actuals=actuals,
        metrics=latest_metrics,
        riskLevels=risk_levels,
        lastUpdatedDate=datetime.utcnow(),
        locationId=region.location_id,
        url=latest_values[CommonFields.CAN_LOCATION_PAGE_URL],
        annotations=build_annotations(one_region, log),
    )


def build_annotations(one_region: OneRegionTimeseriesDataset, log) -> Annotations:
    assert one_region.tag.index.names == [TagField.VARIABLE, TagField.TYPE]
    return Annotations(
        cases=_build_metric_annotations(one_region, CommonFields.CASES, log),
        deaths=_build_metric_annotations(one_region, CommonFields.DEATHS, log),
        positiveTests=_build_metric_annotations(one_region, CommonFields.POSITIVE_TESTS, log),
        negativeTests=_build_metric_annotations(one_region, CommonFields.NEGATIVE_TESTS, log),
        contactTracers=_build_metric_annotations(
            one_region, CommonFields.CONTACT_TRACERS_COUNT, log
        ),
        hospitalBeds=_build_metric_annotations(
            one_region, CommonFields.HOSPITAL_BEDS_IN_USE_ANY, log
        ),
        icuBeds=_build_metric_annotations(one_region, CommonFields.ICU_BEDS, log),
        newCases=_build_metric_annotations(one_region, CommonFields.NEW_CASES, log),
        vaccinesDistributed=_build_metric_annotations(
            one_region, CommonFields.VACCINES_DISTRIBUTED, log
        ),
        vaccinationsInitiated=_build_metric_annotations(
            one_region, CommonFields.VACCINATIONS_INITIATED, log
        ),
        vaccinationsCompleted=_build_metric_annotations(
            one_region, CommonFields.VACCINATIONS_COMPLETED, log
        ),
    )


def _build_metric_annotations(
    tag_series: timeseries.OneRegionTimeseriesDataset, field_name: CommonFields, log
) -> Optional[FieldAnnotations]:

    sources_enum = []
    for source_str in tag_series.provenance.get(field_name, []):
        source_enum = FieldSource.get(source_str)
        if source_enum is None:
            source_enum = FieldSource.OTHER
            log.info(
                METRIC_SOURCES_NOT_FOUND_MESSAGE, field_name=field_name, provenance=source_str,
            )
        sources_enum.append(source_enum)

    anomalies = tag_series.annotations(field_name)
    anomalies = [
        AnomalyAnnotation(
            date=tag.date, original_observation=tag.original_observation, type=tag.type
        )
        for tag in anomalies
    ]

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

    if not sources_enum and not anomalies and not source_url:
        return None

    return FieldAnnotations(sources=sources_enum, anomalies=anomalies, source_url=source_url)


def build_region_timeseries(
    region_summary: RegionSummary,
    timeseries: OneRegionTimeseriesDataset,
    metrics_timeseries: pd.DataFrame,
    risk_level_timeseries: pd.DataFrame,
) -> RegionSummaryWithTimeseries:
    actuals_timeseries = []

    for row in timeseries.yield_records():
        # Timeseries records don't have population
        row[CommonFields.POPULATION] = region_summary.population
        actual = _build_actuals(row).dict()

        # Don't include vaccinations in timeseries before first possible vaccination
        # date to not bloat timeseries.
        if row[CommonFields.DATE] < USA_VACCINATION_START_DATE:
            del actual["vaccinesDistributed"]
            del actual["vaccinationsInitiated"]
            del actual["vaccinationsCompleted"]

        timeseries_row = ActualsTimeseriesRow(**actual, date=row[CommonFields.DATE])
        actuals_timeseries.append(timeseries_row)

    metrics_rows = []
    for metric_row in metrics_timeseries.to_dict(orient="records"):
        # Don't include vaccinations in timeseries before first possible vaccination
        # date to not bloat timeseries.
        if metric_row[CommonFields.DATE] < USA_VACCINATION_START_DATE:
            del metric_row["vaccinationsInitiatedRatio"]
            del metric_row["vaccinationsCompletedRatio"]
        metrics_rows.append(MetricsTimeseriesRow(**metric_row))

    risk_level_rows = [
        RiskLevelTimeseriesRow(**row) for row in risk_level_timeseries.to_dict(orient="records")
    ]
    region_summary_data = {key: getattr(region_summary, key) for (key, _) in region_summary}
    return RegionSummaryWithTimeseries(
        **region_summary_data,
        actualsTimeseries=actuals_timeseries,
        metricsTimeseries=metrics_rows,
        riskLevelsTimeseries=risk_level_rows,
    )


def build_bulk_flattened_timeseries(
    bulk_timeseries: AggregateRegionSummary,
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
        }
        actuals_by_date = {row.date: row for row in region_timeseries.actualsTimeseries}
        metrics_by_date = {row.date: row for row in region_timeseries.metricsTimeseries}
        risk_levels_by_date = {row.date: row for row in region_timeseries.riskLevelsTimeseries}
        dates = sorted({*metrics_by_date.keys(), *actuals_by_date.keys()})
        for date in dates:
            risk_levels = risk_levels_by_date.get(date)
            risk_levels_row = None
            if risk_levels:
                risk_levels_row = RiskLevelsRow(overall=risk_levels.overall)

            data = {
                "date": date,
                "actuals": actuals_by_date.get(date),
                "metrics": metrics_by_date.get(date),
                "riskLevels": risk_levels_row,
            }
            data.update(summary_data)
            row = RegionTimeseriesRowWithHeader(**data)
            rows.append(row)

    return AggregateFlattenedTimeseries(__root__=rows)
