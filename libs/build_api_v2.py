from datetime import datetime
from typing import Optional
import pandas as pd
from api.can_api_v2_definition import (
    Actuals,
    ActualsTimeseriesRow,
    Annotations,
    MetricAnnotations,
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

from api.can_api_v2_definition import MetricAnomaly
from api.can_api_v2_definition import MetricSource
from libs.datasets import timeseries
from libs.datasets.tail_filter import TagField
from libs.datasets.timeseries import OneRegionTimeseriesDataset
from libs.datasets.timeseries import TagType


METRIC_SOURCES_NOT_FOUND_MESSAGE = "Unable to find provenance in MetricSource enum"


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
    assert one_region.tag.index.names == [TagField.VARIABLE, TagField.TYPE, TagField.DATE]
    return Annotations(
        cases=_build_metric_annotations(one_region.tag, CommonFields.CASES, log),
        deaths=_build_metric_annotations(one_region.tag, CommonFields.DEATHS, log),
        positiveTests=_build_metric_annotations(one_region.tag, CommonFields.POSITIVE_TESTS, log),
        negativeTests=_build_metric_annotations(one_region.tag, CommonFields.NEGATIVE_TESTS, log),
        contactTracers=_build_metric_annotations(
            one_region.tag, CommonFields.CONTACT_TRACERS_COUNT, log
        ),
        hospitalBeds=_build_metric_annotations(
            one_region.tag, CommonFields.HOSPITAL_BEDS_IN_USE_ANY, log
        ),
        icuBeds=_build_metric_annotations(one_region.tag, CommonFields.ICU_BEDS, log),
        newCases=_build_metric_annotations(one_region.tag, CommonFields.NEW_CASES, log),
    )


def _build_metric_annotations(
    tag_series: pd.Series, field_name: CommonFields, log
) -> Optional[MetricAnnotations]:
    try:
        metric_tag_df: pd.DataFrame = tag_series[field_name]
    except KeyError:
        return None

    sources_enum = []
    for source_str in metric_tag_df.loc[[TagType.PROVENANCE]]:
        source_enum = MetricSource.get(source_str)
        if source_enum is None:
            source_enum = MetricSource.OTHER
            log.info(
                METRIC_SOURCES_NOT_FOUND_MESSAGE, field_name=field_name, provenance=source_str,
            )
        sources_enum.append(source_enum)

    anomalies_tuples = (
        metric_tag_df.loc[
            [TagType.CUMULATIVE_TAIL_TRUNCATED, TagType.CUMULATIVE_LONG_TAIL_TRUNCATED]
        ]
        .reset_index()
        .itertuples(index=False)
    )

    return MetricAnnotations(
        sources=sources_enum,
        anomalies=[MetricAnomaly(date=t.date, description=t.content) for t in anomalies_tuples],
    )


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
        actual = _build_actuals(row)
        timeseries_row = ActualsTimeseriesRow(**actual.dict(), date=row[CommonFields.DATE])
        actuals_timeseries.append(timeseries_row)

    metrics_rows = [
        MetricsTimeseriesRow(**metric_row)
        for metric_row in metrics_timeseries.to_dict(orient="records")
    ]
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
