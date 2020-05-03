from datetime import datetime, timedelta
from api.can_api_definition import (
    CovidActNowCountiesAPI,
    CovidActNowCountySummary,
    CovidActNowStateSummary,
    CovidActNowAreaSummary,
    CovidActNowCountyTimeseries,
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
from libs.datasets.combined_datasets import build_latest_with_all_fields
from libs.enums import Intervention
from libs.functions import get_can_projection
from libs.datasets.dataset_utils import AggregationLevel
from libs.us_state_abbrev import US_STATE_ABBREV
from libs.datasets import can_model_output_schema as can_schema
from libs.datasets import CovidTrackingDataSource
from libs.datasets import CDSDataset
from libs.datasets.beds import BedsDataset
from libs.build_processed_dataset import get_testing_timeseries_by_state
from libs.build_processed_dataset import get_testing_timeseries_by_fips
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


def _generate_state_actuals(
    projection_row: pd.Series, state_intervention: Intervention, state_beds_data: dict
):
    """Generates Actuals for a state.

    Args:
        projection_row: Output from projection DataFrame.
        state_intervention: Intervention for state
        state_beds_data: Bed data for a specific state.
    """
    latest_values_dataset = build_latest_with_all_fields()
    state = US_STATE_ABBREV[projection_row[rc.STATE_FULL_NAME]]
    latest_actuals = latest_values_dataset.get_data_for_state(state)
    intervention_str = state_intervention.name

    return _Actuals(
        population=latest_actuals[CommonFields.POPULATION],
        intervention=intervention_str,
        cumulativeConfirmedCases=latest_actuals[CommonFields.CASES],
        cumulativeDeaths=latest_actuals[CommonFields.DEATHS],
        cumulativePositiveTests=latest_actuals[CommonFields.POSITIVE_TESTS],
        cumulativeNegativeTests=latest_actuals[CommonFields.NEGATIVE_TESTS],
        hospitalBeds={
            "capacity": latest_actuals[CommonFields.MAX_BED_COUNT] or 0,
            # TODO(chris): Get from assembled sources about current hospitalization data.
            # i.e. NV data we can manually update.
            "currentUsage": None,
            "typicalUsageRate": latest_actuals.get(CommonFields.ALL_BED_TYPICAL_OCCUPANCY_RATE),
        },
        ICUBeds={
            # Note(Chris): We do not currently pass through ICU Bed capacity calculations
            # in the projection_row.  This wouldn't be a ton of work to do, but
            # using the provided beds data for the time being.
            "capacity": latest_actuals[CommonFields.ICU_BEDS],
            "currentUsage": None,
            "typicalUsageRate": latest_actuals[CommonFields.ICU_TYPICAL_OCCUPANCY_RATE],
        },
    )


def _generate_county_actuals(projection_row, state_intervention, county_beds_data):
    intervention_str = state_intervention.name
    latest_values_dataset = build_latest_with_all_fields()
    latest_actuals = latest_values_dataset.get_data_for_fips(projection_row[rc.FIPS])

    icu_beds = None
    if county_beds_data:
        icu_beds = {
            "capacity": latest_actuals[CommonFields.ICU_BEDS],
            "currentUsage": None,
            "typicalUsageRate": latest_actuals[CommonFields.ICU_TYPICAL_OCCUPANCY_RATE],
        }

    return _Actuals(
        population=latest_actuals[CommonFields.POPULATION],
        intervention=intervention_str,
        cumulativeConfirmedCases=latest_actuals[CommonFields.CASES],
        cumulativeDeaths=latest_actuals[CommonFields.DEATHS],
        cumulativePositiveTests=latest_actuals[CommonFields.POSITIVE_TESTS],
        cumulativeNegativeTests=latest_actuals[CommonFields.NEGATIVE_TESTS],
        hospitalBeds={
            "capacity": latest_actuals[CommonFields.MAX_BED_COUNT] or 0,
            "currentUsage": None,
            "typicalUsageRate": latest_actuals.get(CommonFields.ALL_BED_TYPICAL_OCCUPANCY_RATE),
        },
        ICUBeds=icu_beds,
    )


def _generate_state_timeseries_row(json_data_row):

    return CANPredictionTimeseriesRow(
        date=datetime.strptime(json_data_row[can_schema.DATE], "%m/%d/%y"),
        hospitalBedsRequired=json_data_row[can_schema.ALL_HOSPITALIZED],
        hospitalBedCapacity=json_data_row[can_schema.BEDS],
        ICUBedsInUse=json_data_row[can_schema.INFECTED_C],
        ICUBedCapacity=json_data_row[can_schema.ICU_BED_CAPACITY],
        cumulativeDeaths=json_data_row[can_schema.DEAD],
        cumulativeInfected=json_data_row[can_schema.CUMULATIVE_INFECTED],
        ventilatorsInUse=json_data_row[can_schema.CURRENT_VENTILATED],
        ventilatorCapacity=json_data_row[can_schema.VENTILATOR_CAPACITY],
        RtIndicator=json_data_row[can_schema.RT_INDICATOR],
        RtIndicatorCI90=json_data_row[can_schema.RT_INDICATOR_CI90],
        cumulativePositiveTests=_get_or_none(
            json_data_row[CovidTrackingDataSource.Fields.POSITIVE_TESTS]
        ),
        cumulativeNegativeTests=_get_or_none(
            json_data_row[CovidTrackingDataSource.Fields.NEGATIVE_TESTS]
        ),
    )

def _generate_county_actuals_timeseries_row(json_data_row): 
    county_actuals = _generate_county_actuals(json_data_row, state_intervention, )
    return CANActualsTimeseriesRow(json_data_row)

def _generate_county_timeseries_row(json_data_row):
    tested = _get_or_none(json_data_row[CDSDataset.Fields.TESTED])
    cases = _get_or_none(json_data_row[CDSDataset.Fields.CASES])
    negative = tested and cases and (tested - cases)

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
        cumulativeDeaths=json_data_row[can_schema.DEAD],
        cumulativeInfected=json_data_row[can_schema.CUMULATIVE_INFECTED],
        cumulativePositiveTests=cases,
        cumulativeNegativeTests=negative,
    )


def generate_state_timeseries(
    projection_row, intervention, input_dir
) -> CovidActNowStateTimeseries:
    state = US_STATE_ABBREV[projection_row[rc.STATE_FULL_NAME]]
    fips = projection_row[rc.FIPS]
    raw_dataseries = get_can_projection.get_can_raw_data(
        input_dir, state, fips, AggregationLevel.STATE, intervention
    )

    # join in state testing data onto the timeseries
    # left join '%m/%d/%y', so the left join gracefully handles
    # missing state testing data (i.e. NE)
    testing_df = get_testing_timeseries_by_state(state)
    new_df = pd.DataFrame(raw_dataseries).merge(
        testing_df, on="date", how="left"
    )
    can_dataseries = new_df.to_dict(orient="records")
    bed_data = get_can_projection.get_beds_data()
    state_bed_data = bed_data.get_data_for_state(state)

    timeseries = []
    for data_series in can_dataseries:
        timeseries.append(_generate_state_timeseries_row(data_series))
    projections = _generate_api_for_projections(projection_row)
    if len(timeseries) < 1:
        raise Exception(f"State time series empty for {intervention.name}")

    state_intervention = get_can_projection.get_intervention_for_state(state)

    return CovidActNowStateTimeseries(
        lat=projection_row[rc.LATITUDE],
        long=projection_row[rc.LONGITUDE],
        actuals=_generate_state_actuals(projection_row, state_intervention, state_bed_data),
        stateName=projection_row[rc.STATE_FULL_NAME],
        fips=projection_row[rc.FIPS],
        lastUpdatedDate=_format_date(projection_row[rc.LAST_UPDATED]),
        projections=projections,
        timeseries=timeseries,
    )


def generate_county_timeseries(projection_row, intervention, input_dir):
    state_abbrev = US_STATE_ABBREV[projection_row[rc.STATE_FULL_NAME]]
    fips = projection_row[rc.FIPS]

    raw_dataseries = get_can_projection.get_can_raw_data(
        input_dir, state_abbrev, fips, AggregationLevel.COUNTY, intervention
    )

    testing_df = get_testing_timeseries_by_fips(fips)
    new_df = pd.DataFrame(raw_dataseries).merge(
        testing_df, on="date", how="left"
    )

    can_dataseries = new_df.to_dict(orient="records")

    timeseries = []
    for data_series in can_dataseries:
        timeseries.append(_generate_county_timeseries_row(data_series))
    if len(timeseries) < 1:
        raise Exception(f"County time series empty for {intervention.name}")
    projections = _generate_api_for_projections(projection_row)
    state_intervention = get_can_projection.get_intervention_for_state(state_abbrev)
    bed_data = get_can_projection.get_beds_data()
    county_bed_data = bed_data.get_data_for_fips(fips)
    return CovidActNowCountyTimeseries(
        lat=projection_row[rc.LATITUDE],
        long=projection_row[rc.LONGITUDE],
        actuals=_generate_county_actuals(projection_row, state_intervention, county_bed_data),
        stateName=projection_row[rc.STATE_FULL_NAME],
        countyName=projection_row[rc.COUNTY],
        fips=projection_row[rc.FIPS],
        lastUpdatedDate=_format_date(projection_row[rc.LAST_UPDATED]),
        projections=projections,
        timeseries=timeseries,
    )


def generate_api_for_state_projection_row(projection_row) -> CovidActNowStateSummary:
    state_abbrev = US_STATE_ABBREV[projection_row[rc.STATE_FULL_NAME]]
    projections = _generate_api_for_projections(projection_row)
    state_intervention = get_can_projection.get_intervention_for_state(state_abbrev)
    bed_data = get_can_projection.get_beds_data()
    state_bed_data = bed_data.get_data_for_state(state_abbrev)
    state_result = CovidActNowStateSummary(
        lat=projection_row[rc.LATITUDE],
        long=projection_row[rc.LONGITUDE],
        actuals=_generate_state_actuals(projection_row, state_intervention, state_bed_data),
        stateName=projection_row[rc.STATE_FULL_NAME],
        fips=projection_row[rc.FIPS],
        lastUpdatedDate=_format_date(projection_row[rc.LAST_UPDATED]),
        projections=projections,
    )
    return state_result


def generate_api_for_county_projection_row(projection_row):
    state_abbrev = US_STATE_ABBREV[projection_row[rc.STATE_FULL_NAME]]
    projections = _generate_api_for_projections(projection_row)
    state_intervention = get_can_projection.get_intervention_for_state(state_abbrev)
    fips = projection_row[rc.FIPS]
    bed_data = get_can_projection.get_beds_data()
    county_bed_data = bed_data.get_data_for_fips(fips)

    county_result = CovidActNowCountySummary(
        lat=projection_row[rc.LATITUDE],
        long=projection_row[rc.LONGITUDE],
        actuals=_generate_county_actuals(projection_row, state_intervention, county_bed_data),
        stateName=projection_row[rc.STATE_FULL_NAME],
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
