from datetime import datetime
from typing import Optional
import pandas as pd
from api.can_api_v2_definition import (
    Actuals,
    ActualsTimeseriesRow,
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
from libs.datasets.timeseries import OneRegionTimeseriesDataset
from libs import pipeline


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
        vaccinesDistributed=actual_data[CommonFields.VACCINES_DISTRIBUTED],
        vaccinationsInitiated=actual_data[CommonFields.VACCINATIONS_INITIATED],
        # Vaccinations completed currently optional as data is not yet flowing through.
        # This will allow us to include vaccines completed data as soon as its scraped.
        vaccinationsCompleted=actual_data.get(CommonFields.VACCINATIONS_COMPLETED),
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
