from typing import Optional
from datetime import datetime, timedelta
from api.can_api_definition import (
    CovidActNowBulkSummary,
    CovidActNowAreaSummary,
    CovidActNowAreaTimeseries,
    CovidActNowBulkFlattenedTimeseries,
    PredictionTimeseriesRowWithHeader,
    CANPredictionTimeseriesRow,
    CANActualsTimeseriesRow,
    _Projections,
    _Actuals,
    _ResourceUsageProjection,
)
from covidactnow.datapublic.common_fields import CommonFields
from libs.enums import Intervention
from libs.functions import get_can_projection
from libs import us_state_abbrev
from libs.datasets import can_model_output_schema as can_schema
from libs.datasets.sources.can_pyseir_location_output import CANPyseirLocationOutput
from libs.datasets.timeseries import TimeseriesDataset
from libs.datasets.latest_values_dataset import LatestValuesDataset
import pandas as pd


def _generate_api_for_projections(model_output: CANPyseirLocationOutput):
    _hospital_beds = _ResourceUsageProjection(
        peakDate=model_output.peak_hospitalizations_date,
        shortageStartDate=model_output.hospitals_shortfall_date,
        peakShortfall=model_output.peak_hospitalizations_shortfall,
    )
    projections = _Projections(
        totalHospitalBeds=_hospital_beds,
        ICUBeds=None,
        Rt=model_output.latest_rt,
        RtCI90=model_output.latest_rt_ci90,
    )
    return projections


def _generate_actuals(actual_data: dict, intervention: Intervention) -> _Actuals:
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

    return _Actuals(
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


def _generate_prediction_timeseries_row(json_data_row) -> CANPredictionTimeseriesRow:

    return CANPredictionTimeseriesRow(
        date=json_data_row[can_schema.DATE].to_pydatetime(),
        hospitalBedsRequired=json_data_row[can_schema.ALL_HOSPITALIZED],
        hospitalBedCapacity=json_data_row[can_schema.BEDS],
        ICUBedsInUse=json_data_row[can_schema.INFECTED_C],
        ICUBedCapacity=json_data_row[can_schema.ICU_BED_CAPACITY],
        ventilatorsInUse=json_data_row[can_schema.CURRENT_VENTILATED],
        ventilatorCapacity=json_data_row[can_schema.VENTILATOR_CAPACITY],
        RtIndicator=json_data_row[can_schema.RT_INDICATOR],
        RtIndicatorCI90=json_data_row[can_schema.RT_INDICATOR_CI90],
        currentInfected=json_data_row[can_schema.ALL_INFECTED],
        currentSusceptible=json_data_row[can_schema.TOTAL_SUSCEPTIBLE],
        currentExposed=json_data_row[can_schema.EXPOSED],
        cumulativeDeaths=json_data_row[can_schema.DEAD],
        cumulativeInfected=json_data_row[can_schema.CUMULATIVE_INFECTED],
    )


def generate_area_summary(
    latest_values: dict, model_output: Optional[CANPyseirLocationOutput],
) -> CovidActNowAreaSummary:
    fips = latest_values[CommonFields.FIPS]
    state = latest_values[CommonFields.STATE]
    state_intervention = get_can_projection.get_intervention_for_state(state)

    actuals = _generate_actuals(latest_values, state_intervention)

    projections = None
    if model_output:
        projections = _generate_api_for_projections(model_output)

    return CovidActNowAreaSummary(
        population=latest_values[CommonFields.POPULATION],
        stateName=us_state_abbrev.ABBREV_US_STATE[state],
        countyName=latest_values.get(CommonFields.COUNTY),
        fips=fips,
        lat=latest_values.get(CommonFields.LATITUDE),
        long=latest_values.get(CommonFields.LONGITUDE),
        actuals=actuals,
        # TODO(chris): change this to reflect latest time data updated?
        lastUpdatedDate=datetime.utcnow(),
        projections=projections,
    )


def generate_area_timeseries(
    area_summary: CovidActNowAreaSummary,
    timeseries: TimeseriesDataset,
    model_output: Optional[CANPyseirLocationOutput],
) -> CovidActNowAreaTimeseries:
    if not area_summary.intervention:
        # All area summaries here are expected to have actuals values.
        # It's a bit unclear why the actuals value is optional in the first place,
        # but at this point we expect actuals to have been included.
        raise AssertionError("Area summary missing actuals")

    actuals_timeseries = []

    for row in timeseries.records:
        # Timeseries records don't have population
        row[CommonFields.POPULATION] = area_summary.population
        actual = _generate_actuals(row, area_summary.intervention,)
        timeseries_row = CANActualsTimeseriesRow(**actual.dict(), date=row[CommonFields.DATE])
        actuals_timeseries.append(timeseries_row)

    model_timeseries = []
    if model_output:
        model_timeseries = [
            _generate_prediction_timeseries_row(row)
            for row in model_output.data.to_dict(orient="records")
        ]

    area_summary_data = {key: getattr(area_summary, key) for (key, _) in area_summary}
    return CovidActNowAreaTimeseries(
        **area_summary_data, timeseries=model_timeseries, actualsTimeseries=actuals_timeseries
    )


def generate_bulk_flattened_timeseries(
    bulk_timeseries: CovidActNowBulkSummary,
) -> CovidActNowBulkFlattenedTimeseries:
    rows = []
    for area_timeseries in bulk_timeseries.__root__:
        # Iterate through each state or county in data, adding summary data to each
        # timeseries row.
        summary_data = {
            "countryName": area_timeseries.countryName,
            "countyName": area_timeseries.countyName,
            "stateName": area_timeseries.stateName,
            "fips": area_timeseries.fips,
            "lat": area_timeseries.lat,
            "long": area_timeseries.long,
            "intervention": area_timeseries.intervention.name,
            # TODO(chris): change this to reflect latest time data updated?
            "lastUpdatedDate": datetime.utcnow(),
        }

        for timeseries_data in area_timeseries.timeseries:
            timeseries_row = PredictionTimeseriesRowWithHeader(
                **summary_data, **timeseries_data.dict()
            )
            rows.append(timeseries_row)

    return CovidActNowBulkFlattenedTimeseries(__root__=rows)
