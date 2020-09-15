from datetime import datetime
from typing import Optional

from api.can_api_v2_definition import (
    Actuals,
    ActualsTimeseriesRow,
    AggregateFlattenedTimeseries,
    AggregateRegionSummary,
    Metrics,
    RegionSummary,
    RegionSummaryWithTimeseries,
)
from covidactnow.datapublic.common_fields import CommonFields
from libs import us_state_abbrev
from libs.datasets import can_model_output_schema as can_schema
from libs.datasets.sources.can_pyseir_location_output import CANPyseirLocationOutput
from libs.datasets.timeseries import TimeseriesDataset
from libs.enums import Intervention


def _generate_actuals(actual_data: dict) -> Actuals:
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
    )


def generate_region_summary(
    latest_values: dict, latest_metrics: Optional[Metrics],
) -> RegionSummary:
    actuals = _generate_actuals(latest_values)

    return RegionSummary(
        fips=latest_values[CommonFields.FIPS],
        country=latest_values.get(CommonFields.COUNTRY),
        state=latest_values[CommonFields.STATE],
        county=latest_values.get(CommonFields.COUNTY),
        level=latest_values[CommonFields.AGGREGATE_LEVEL],
        lat=latest_values.get(CommonFields.LATITUDE),
        long=latest_values.get(CommonFields.LONGITUDE),
        population=latest_values[CommonFields.POPULATION],
        actuals=actuals,
        metrics=latest_metrics,
        lastUpdatedDate=datetime.utcnow(),
    )


def generate_region_timeseries(
    region_summary: RegionSummary, timeseries: TimeseriesDataset, metrics_timeseries,
) -> RegionSummaryWithTimeseries:
    actuals_timeseries = []

    for row in timeseries.yield_records():
        # Timeseries records don't have population
        row[CommonFields.POPULATION] = region_summary.population
        actual = _generate_actuals(row)
        timeseries_row = ActualsTimeseriesRow(**actual.dict(), date=row[CommonFields.DATE])
        actuals_timeseries.append(timeseries_row)

    region_summary_data = {key: getattr(region_summary, key) for (key, _) in region_summary}
    return RegionSummaryWithTimeseries(
        **region_summary_data,
        actualsTimeseries=actuals_timeseries,
        metricsTimeseries=metrics_timeseries
    )


def generate_bulk_flattened_timeseries(
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
            "lastUpdatedDate": datetime.utcnow(),
        }

    return AggregateFlattenedTimeseries(__root__=rows)
