from typing import Optional
from datetime import datetime, timedelta
from api.can_api_definition import (
    CovidActNowCountiesAPI,
    CovidActNowCountySummary,
    CovidActNowStateSummary,
    CovidActNowAreaSummary,
    CovidActNowCountyTimeseries,
    CovidActNowAreaTimeseries,
    CovidActNowStateTimeseries,
    CANPredictionTimeseriesRow,
    CANActualsTimeseriesRow,
    _Projections,
    _Actuals,
    _ResourceUsageProjection,
)
from libs.constants import NULL_VALUE
from libs.datasets import results_schema as rc
from libs.datasets.common_fields import CommonFields
from libs.datasets import combined_datasets
from libs.enums import Intervention
from libs.functions import get_can_projection
from libs.datasets.dataset_utils import AggregationLevel
from libs.us_state_abbrev import US_STATE_ABBREV
from libs import us_state_abbrev
from libs.datasets import can_model_output_schema as can_schema
from libs.datasets import CovidTrackingDataSource
from libs.datasets import CDSDataset
from libs.datasets.sources.can_pyseir_location_output import CANPyseirLocationOutput
from libs.datasets.timeseries import TimeseriesDataset
from libs.datasets.latest_values_dataset import LatestValuesDataset
from libs.build_processed_dataset import get_testing_timeseries_by_state
from libs.build_processed_dataset import get_testing_timeseries_by_fips
import pandas as pd


def _format_date(input_date):
    if not input_date:
        raise Exception("Can't format a date that doesn't exist")
    if isinstance(input_date, str):
        # note if this is already in iso format it will be grumpy. maybe use dateutil
        datetime_obj = datetime.strptime(input_date, "%m/%d/%Y %H:%M")
        return datetime_obj
    if isinstance(input_date, datetime):
        return input_date
    raise Exception("Invalid date type when converting to api")


def _get_date_or_none(panda_date_or_none):
    """ Projection Null value is a string NULL so if this date value is a string,
     make it none. Otherwise convert to the python datetime. Example
     of this being null is when there is no bed shortfall, the shortfall dates is none """
    if isinstance(panda_date_or_none, str):
        return None
    return panda_date_or_none.to_pydatetime()


def _get_or_none(value):
    if isinstance(value, str) and value == NULL_VALUE:
        return None
    elif pd.isna(value):
        return None
    else:
        return value


def _get_or_zero(value):
    if isinstance(value, str) and value == NULL_VALUE:
        return 0
    else:
        return value


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


def _generate_actuals_timeseries(actuals_timeseries_dataset, intervention):
    actual_timeseries_api_response = []
    for row in actuals_timeseries_dataset:
        actual = _generate_actuals(row, intervention)
        timeseries_actual = CANActualsTimeseriesRow(**actual.dict(), date=row[CommonFields.DATE])
        actual_timeseries_api_response.append(timeseries_actual)
    return actual_timeseries_api_response


def _generate_prediction_timeseries_row(json_data_row) -> CANPredictionTimeseriesRow:

    return CANPredictionTimeseriesRow(
        date=datetime.strptime(json_data_row[can_schema.DATE], "%m/%d/%y"),
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
        # TODO: Either deprecate this field or figure out how to pass test data through.
        # cumulativePositiveTests=cases,
        # cumulativeNegativeTests=negative,
    )


def generate_area_summary(
    fips: str, intervention: Intervention, latest_values: dict, projection_row: Optional[dict],
):
    state = latest_values[CommonFields.STATE]
    state_intervention = get_can_projection.get_intervention_for_state(state)
    actuals = _generate_actuals(latest_values, state_intervention)

    projections = None
    if projection_row:
        projections = _generate_api_for_projections(projection_row)

    return CovidActNowAreaSummary(
        population=latest_values[CommonFields.POPULATION],
        stateName=us_state_abbrev.ABBREV_US_STATE[state],
        countyName=latest_values.get(CommonFields.COUNTY),
        fips=fips,
        lat=latest_values.get(CommonFields.LATITUDE),
        long=latest_values.get(CommonFields.LONGITUDE),
        actuals=actuals,
        lastUpdatedDate=datetime.utcnow(),
        projections=projections,
    )


def generate_area_timeseries(
    area_summary: CovidActNowAreaSummary,
    timeseries: TimeseriesDataset,
    model_timeseries: pd.DataFrame,
) -> CovidActNowAreaTimeseries:
    if not area_summary.intervention:
        # All area summaries here are expected to have actuals values.
        # It's a bit unclear why the actuals value is optional in the first place,
        # but at this point we expect actuals to have been included.
        raise AssertionError("Area summary missing actuals")

    actuals_timeseries = []

    for row in timeseries.records:
        actual = _generate_actuals(row, area_summary.intervention)
        timeseries_row = CANActualsTimeseriesRow(**actual.dict(), date=row[CommonFields.DATE])
        actuals_timeseries.append(timeseries_actual)

    model_timeseries = [
        _generate_prediction_timeseries_row(row)
        for row in model_timeseries.to_dict(orient="records")
    ]

    area_summary_data = {key: getattr(area_summary, key) for (key, _) in CovidActNowAreaSummary}
    return CovidActNowAreaTimeseries(
        **area_summary_data, timeseries=model_timeseries, actualsTimeseries=actuals_timeseries
    )
