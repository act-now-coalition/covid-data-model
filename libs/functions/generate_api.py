from datetime import datetime, timedelta
from api.can_api_definition import (
    CovidActNowCountiesAPI,
    CovidActNowCountySummary,
    CovidActNowStateSummary,
    CovidActNowAreaSummary,
    CovidActNowCountyTimeseries,
    CovidActNowStateTimeseries,
    CANPredictionTimeseriesRow,
    _Projections,
    _Actuals,
    _ResourceUsageProjection,
)
from libs.constants import NULL_VALUE
from libs.datasets import results_schema as rc
from libs.enums import Intervention
from libs.functions import get_can_projection
from libs.datasets.dataset_utils import AggregationLevel
from libs.us_state_abbrev import US_STATE_ABBREV
from libs.datasets import can_model_output_schema as can_schema
from libs.datasets import CovidTrackingDataSource
from libs.build_processed_dataset import get_testing_timeseries_by_state
import pandas as pd

FRAMES = 32
DAYS_PER_FRAME = 4


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


def _generate_api_for_projections(projection_row):
    peak_date = _get_date_or_none(projection_row[rc.PEAK_HOSPITALIZATIONS])
    shortage_start_date = _get_date_or_none(projection_row[rc.HOSPITAL_SHORTFALL_DATE])
    _hospital_beds = _ResourceUsageProjection(
        peakDate=_get_or_none(peak_date),
        shortageStartDate=shortage_start_date,
        peakShortfall=_get_or_zero(projection_row[rc.PEAK_HOSPITALIZATION_SHORTFALL]),
    )
    projections = _Projections(
        totalHospitalBeds=_hospital_beds,
        ICUBeds=None,
        Rt=_get_or_zero(projection_row[rc.RT]),
        RtCI90=_get_or_zero(projection_row[rc.RT_CI90]),
    )
    return projections

def _generate_state_actuals(projection_row, state):
    intervention_str = get_can_projection.get_intervention_for_state(state).name

    return _Actuals(
        population=projection_row[rc.POPULATION],
        intervention=intervention_str,
        cumulativeConfirmedCases=projection_row[rc.CURRENT_CONFIRMED],
        cumulativeDeaths=projection_row[rc.CURRENT_DEATHS],
        cumulativePositiveTests=_get_or_none(projection_row[rc.CUMULATIVE_POSITIVE_TESTS]),
        cumulativeNegativeTests=_get_or_none(projection_row[rc.CUMULATIVE_NEGATIVE_TESTS]),
        hospitalBeds={
            "capacity": projection_row[rc.PEAK_BED_CAPACITY],
            "currentUsage": None # TODO(igor): Get from Covidtracking source
        },
        ICUBeds = None
    )

def _generate_county_actuals(projection_row, state):
    intervention_str = get_can_projection.get_intervention_for_state(state).name
    return _Actuals(
        population=projection_row[rc.POPULATION],
        intervention=intervention_str,
        cumulativeConfirmedCases=projection_row[rc.CURRENT_CONFIRMED],
        cumulativeDeaths=projection_row[rc.CURRENT_DEATHS],
        cumulativePositiveTests=None,
        cumulativeNegativeTests=None,
        hospitalBeds = {
            "capacity": projection_row[rc.PEAK_BED_CAPACITY],
            "currentUsage": None # TODO(igor): Get from Covidtracking source
        },
        ICUBeds = None
    )

def _generate_state_timeseries_row(json_data_row):

    return CANPredictionTimeseriesRow(
        date=datetime.strptime(json_data_row[can_schema.DATE], "%m/%d/%y"),
        hospitalBedsRequired=json_data_row[can_schema.ALL_HOSPITALIZED],
        hospitalBedCapacity=json_data_row[can_schema.BEDS],
        ICUBedsInUse=json_data_row[can_schema.INFECTED_C],
        ICUBedCapacity=None,
        cumulativeDeaths=json_data_row[can_schema.DEAD],
        cumulativeInfected=json_data_row[can_schema.CUMULATIVE_INFECTED],
        cumulativePositiveTests=_get_or_none(json_data_row[CovidTrackingDataSource.Fields.POSITIVE_TESTS]),
        cumulativeNegativeTests=_get_or_none(json_data_row[CovidTrackingDataSource.Fields.NEGATIVE_TESTS]),
    )

def _generate_county_timeseries_row(json_data_row):
    return CANPredictionTimeseriesRow(
        date=datetime.strptime(json_data_row[can_schema.DATE], "%m/%d/%y"),
        hospitalBedsRequired=json_data_row[can_schema.ALL_HOSPITALIZED],
        hospitalBedCapacity=json_data_row[can_schema.BEDS],
        ICUBedsInUse=json_data_row[can_schema.INFECTED_C],
        ICUBedCapacity=None,
        cumulativeDeaths=json_data_row[can_schema.DEAD],
        cumulativeInfected=json_data_row[can_schema.CUMULATIVE_INFECTED],
        cumulativePositiveTests=None,
        cumulativeNegativeTests=None,
    )


def generate_state_timeseries(projection_row, intervention, input_dir) -> CovidActNowStateTimeseries:
    state_abbrev = US_STATE_ABBREV[projection_row[rc.STATE]]
    fips = projection_row[rc.FIPS]
    raw_dataseries = get_can_projection.get_can_raw_data(
        input_dir, state_abbrev, fips, AggregationLevel.STATE, intervention
    )

    # join in state testing data onto the timeseries
    # left join '%m/%d/%y', so the left join gracefully handles missing state testing data (i.e. NE)
    testing_df = get_testing_timeseries_by_state(projection_row[rc.STATE])
    new_df = pd.DataFrame(raw_dataseries).merge(
        testing_df, left_on='date', right_on='date', how='left')

    # reformat as dict
    can_dataseries = new_df.to_dict(orient='records')

    timeseries = []
    for data_series in can_dataseries:
        timeseries.append(_generate_state_timeseries_row(data_series))
    projections = _generate_api_for_projections(projection_row)
    if len(timeseries) < 1:
        raise Exception(f"State time series empty for {intervention.name}")

    return CovidActNowStateTimeseries(
        lat=projection_row[rc.LATITUDE],
        long=projection_row[rc.LONGITUDE],
        actuals=_generate_state_actuals(projection_row, state_abbrev),
        stateName=projection_row[rc.STATE],
        fips=projection_row[rc.FIPS],
        lastUpdatedDate=_format_date(projection_row[rc.LAST_UPDATED]),
        projections=projections,
        timeseries=timeseries,
    )


def generate_county_timeseries(projection_row, intervention, input_dir):
    state_abbrev = US_STATE_ABBREV[projection_row[rc.STATE]]
    fips = projection_row[rc.FIPS]
    can_dataseries = get_can_projection.get_can_raw_data(
        input_dir, state_abbrev, fips, AggregationLevel.COUNTY, intervention
    )

    timeseries = []
    for data_series in can_dataseries:
        timeseries.append(_generate_county_timeseries_row(data_series))
    if len(timeseries) < 1:
        raise Exception(f"County time series empty for {intervention.name}")
    projections = _generate_api_for_projections(projection_row)
    return CovidActNowCountyTimeseries(
        lat=projection_row[rc.LATITUDE],
        long=projection_row[rc.LONGITUDE],
        actuals=_generate_county_actuals(projection_row, state_abbrev),
        stateName=projection_row[rc.STATE],
        countyName=projection_row[rc.COUNTY],
        fips=projection_row[rc.FIPS],
        lastUpdatedDate=_format_date(projection_row[rc.LAST_UPDATED]),
        projections=projections,
        timeseries=timeseries,
    )


def generate_api_for_state_projection_row(projection_row) -> CovidActNowStateSummary:
    state_abbrev = US_STATE_ABBREV[projection_row[rc.STATE]]
    projections = _generate_api_for_projections(projection_row)

    state_result = CovidActNowStateSummary(
        lat=projection_row[rc.LATITUDE],
        long=projection_row[rc.LONGITUDE],
        actuals=_generate_state_actuals(projection_row, state_abbrev),
        stateName=projection_row[rc.STATE],
        fips=projection_row[rc.FIPS],
        lastUpdatedDate=_format_date(projection_row[rc.LAST_UPDATED]),
        projections=projections,
    )
    return state_result


def generate_api_for_county_projection_row(projection_row):
    state_abbrev = US_STATE_ABBREV[projection_row[rc.STATE]]
    projections = _generate_api_for_projections(projection_row)

    county_result = CovidActNowCountySummary(
        lat=projection_row[rc.LATITUDE],
        long=projection_row[rc.LONGITUDE],
        actuals=_generate_county_actuals(projection_row, state_abbrev),
        stateName=projection_row[rc.STATE],
        countyName=projection_row[rc.COUNTY],
        fips=projection_row[rc.FIPS],
        lastUpdatedDate=_format_date(projection_row[rc.LAST_UPDATED]),
        projections=projections,
    )
    return county_result


def generate_api_for_county_projection(projection) -> CovidActNowCountiesAPI:
    api_results = []

    for index, county_row in projection.iterrows():
        county_result = generate_api_for_county_projection_row(county_row)
        api_results.append(county_result)
    return CovidActNowCountiesAPI(__root__=api_results)
