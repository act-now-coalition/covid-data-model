from datetime import datetime
from typing import Optional

from api.can_api_definition import (
    Actuals,
    ActualsTimeseriesRow,
    AggregateFlattenedTimeseries,
    AggregateRegionSummary,
    Metrics,
    PredictionTimeseriesRowWithHeader,
    RegionSummary,
    RegionSummaryWithTimeseries,
)
from covidactnow.datapublic.common_fields import CommonFields
from libs import us_state_abbrev
from libs.datasets.timeseries import OneRegionTimeseriesDataset
from libs.enums import Intervention
from libs.functions import get_can_projection


def _generate_actuals(actual_data: dict, intervention: Intervention) -> Actuals:
    """Generate actuals entry.

    Args:
        actual_data: Dictionary of data, generally derived one of the combined datasets.
        intervention: Current state level intervention.

    """
    total_bed_capacity = actual_data.get(CommonFields.MAX_BED_COUNT)
    typical_usage_rate = actual_data.get(CommonFields.ALL_BED_TYPICAL_OCCUPANCY_RATE)
    capacity = None
    if total_bed_capacity and typical_usage_rate:
        # At the dawn of the API, the capacity for hospital beds actually referred to the
        # expected bed capacity available for covid patients. We calculated this
        # by multiplying remaining capacity by total beds available multiplied by a
        # scale factor meant to represent the ratio of beds expected to become available
        # as a result of less hospital utilization.
        capacity = (1 - typical_usage_rate) * total_bed_capacity * 2.07

    return Actuals(
        population=actual_data.get(CommonFields.POPULATION),
        intervention=intervention.name,
        cumulativeConfirmedCases=actual_data[CommonFields.CASES],
        cumulativeDeaths=actual_data[CommonFields.DEATHS],
        cumulativePositiveTests=actual_data.get(CommonFields.POSITIVE_TESTS),
        cumulativeNegativeTests=actual_data.get(CommonFields.NEGATIVE_TESTS),
        hospitalBeds={
            "capacity": capacity,
            "totalCapacity": total_bed_capacity,
            "currentUsageCovid": actual_data.get(CommonFields.CURRENT_HOSPITALIZED),
            "currentUsageTotal": actual_data.get(CommonFields.CURRENT_HOSPITALIZED_TOTAL),
            "typicalUsageRate": typical_usage_rate,
        },
        ICUBeds={
            "capacity": actual_data.get(CommonFields.ICU_BEDS),
            "totalCapacity": actual_data.get(CommonFields.ICU_BEDS),
            "currentUsageCovid": actual_data.get(CommonFields.CURRENT_ICU),
            "currentUsageTotal": actual_data.get(CommonFields.CURRENT_ICU_TOTAL),
            "typicalUsageRate": actual_data.get(CommonFields.ICU_TYPICAL_OCCUPANCY_RATE),
        },
        contactTracers=actual_data.get(CommonFields.CONTACT_TRACERS_COUNT),
    )


def generate_region_summary(
    latest_values: dict, latest_metrics: Optional[Metrics],
) -> RegionSummary:
    fips = latest_values[CommonFields.FIPS]
    state = latest_values[CommonFields.STATE]
    state_intervention = get_can_projection.get_intervention_for_state(state)

    actuals = _generate_actuals(latest_values, state_intervention)

    projections = None

    return RegionSummary(
        population=latest_values[CommonFields.POPULATION],
        stateName=us_state_abbrev.ABBREV_US_STATE[state],
        countyName=latest_values.get(CommonFields.COUNTY),
        fips=fips,
        lat=latest_values.get(CommonFields.LATITUDE),
        long=latest_values.get(CommonFields.LONGITUDE),
        actuals=actuals,
        metrics=latest_metrics,
        # TODO(chris): change this to reflect latest time data updated?
        lastUpdatedDate=datetime.utcnow(),
        projections=projections,
    )


def generate_region_timeseries(
    region_summary: RegionSummary, timeseries: OneRegionTimeseriesDataset, metrics_timeseries,
) -> RegionSummaryWithTimeseries:
    if not region_summary.intervention:
        # All region summaries here are expected to have actuals values.
        # It's a bit unclear why the actuals value is optional in the first place,
        # but at this point we expect actuals to have been included.
        raise AssertionError("Region summary missing actuals")

    actuals_timeseries = []

    for row in timeseries.yield_records():
        # Timeseries records don't have population
        row[CommonFields.POPULATION] = region_summary.population
        actual = _generate_actuals(row, region_summary.intervention)
        timeseries_row = ActualsTimeseriesRow(**actual.dict(), date=row[CommonFields.DATE])
        actuals_timeseries.append(timeseries_row)

    model_timeseries = []

    region_summary_data = {key: getattr(region_summary, key) for (key, _) in region_summary}
    return RegionSummaryWithTimeseries(
        **region_summary_data,
        timeseries=model_timeseries,
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
            "countryName": region_timeseries.countryName,
            "countyName": region_timeseries.countyName,
            "stateName": region_timeseries.stateName,
            "fips": region_timeseries.fips,
            "lat": region_timeseries.lat,
            "long": region_timeseries.long,
            "intervention": region_timeseries.intervention.name,
            # TODO(chris): change this to reflect latest time data updated?
            "lastUpdatedDate": datetime.utcnow(),
        }

        for timeseries_data in region_timeseries.timeseries:
            timeseries_row = PredictionTimeseriesRowWithHeader(
                **summary_data, **timeseries_data.dict()
            )
            rows.append(timeseries_row)

    return AggregateFlattenedTimeseries(__root__=rows)
