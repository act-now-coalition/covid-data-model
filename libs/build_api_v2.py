from datetime import datetime
from typing import Optional

from api.can_api_v2_definition import (
    Actuals,
    ActualsTimeseriesRow,
    AggregateFlattenedTimeseries,
    AggregateRegionSummary,
    Metrics,
    RiskLevels,
    RegionSummary,
    RegionSummaryWithTimeseries,
    RegionTimeseriesRowWithHeader,
)
from covidactnow.datapublic.common_fields import CommonFields
from libs.datasets.timeseries import OneRegionTimeseriesDataset
from libs import pipeline


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
            "currentUsageTotal": actual_data.get(CommonFields.CURRENT_HOSPITALIZED_TOTAL),
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
    latest_values: dict,
    latest_metrics: Optional[Metrics],
    risk_levels: RiskLevels,
    region: pipeline.Region,
) -> RegionSummary:

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
    )


def build_region_timeseries(
    region_summary: RegionSummary, timeseries: OneRegionTimeseriesDataset, metrics_timeseries,
) -> RegionSummaryWithTimeseries:
    actuals_timeseries = []

    for row in timeseries.yield_records():
        # Timeseries records don't have population
        row[CommonFields.POPULATION] = region_summary.population
        actual = _build_actuals(row)
        timeseries_row = ActualsTimeseriesRow(**actual.dict(), date=row[CommonFields.DATE])
        actuals_timeseries.append(timeseries_row)

    region_summary_data = {key: getattr(region_summary, key) for (key, _) in region_summary}
    return RegionSummaryWithTimeseries(
        **region_summary_data,
        actualsTimeseries=actuals_timeseries,
        metricsTimeseries=metrics_timeseries
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
        dates = sorted({*metrics_by_date.keys(), *actuals_by_date.keys()})
        for date in dates:
            data = {
                "date": date,
                "actuals": actuals_by_date.get(date),
                "metrics": metrics_by_date.get(date),
            }
            data.update(summary_data)
            row = RegionTimeseriesRowWithHeader(**data)
            rows.append(row)

    return AggregateFlattenedTimeseries(__root__=rows)
