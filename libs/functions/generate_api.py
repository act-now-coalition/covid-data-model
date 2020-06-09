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
from libs.datasets import combined_datasets
from libs.enums import Intervention
from libs.functions import get_can_projection
from libs.datasets.dataset_utils import AggregationLevel
from libs.us_state_abbrev import US_STATE_ABBREV
from libs.datasets import can_model_output_schema as can_schema
from libs.datasets import CovidTrackingDataSource
from libs.datasets import CDSDataset
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


def _generate_actuals(actual_data, intervention) -> _Actuals:

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


def _generate_state_timeseries_row(json_data_row):

    return CANPredictionTimeseriesRow(
        date=datetime.strptime(json_data_row[can_schema.DATE], "%m/%d/%y"),
        hospitalBedsRequired=json_data_row[can_schema.ALL_HOSPITALIZED],
        hospitalBedCapacity=json_data_row[can_schema.BEDS],
        ICUBedsInUse=json_data_row[can_schema.INFECTED_C],
        ICUBedCapacity=json_data_row[can_schema.ICU_BED_CAPACITY],
        cumulativeDeaths=json_data_row[can_schema.DEAD],
        cumulativeInfected=json_data_row[can_schema.CUMULATIVE_INFECTED],
        currentInfected=json_data_row[can_schema.ALL_INFECTED],
        currentSusceptible=json_data_row[can_schema.TOTAL_SUSCEPTIBLE],
        currentExposed=json_data_row[can_schema.EXPOSED],
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
        currentInfected=json_data_row[can_schema.ALL_INFECTED],
        currentSusceptible=json_data_row[can_schema.TOTAL_SUSCEPTIBLE],
        currentExposed=json_data_row[can_schema.EXPOSED],
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
    new_df = pd.DataFrame(raw_dataseries).merge(testing_df, on="date", how="left")
    can_dataseries = new_df.to_dict(orient="records")

    timeseries = []
    for data_series in can_dataseries:
        timeseries.append(_generate_state_timeseries_row(data_series))

    projections = _generate_api_for_projections(projection_row)
    if len(timeseries) < 1:
        raise Exception(f"State time series empty for {intervention.name}")

    state_intervention = get_can_projection.get_intervention_for_state(state)
    actuals_ts = combined_datasets.build_us_timeseries_with_all_fields()
    actual_latest = combined_datasets.build_us_latest_with_all_fields()
    state_latest = actual_latest.get_record_for_state(state)

    return CovidActNowStateTimeseries(
        population=state_latest[CommonFields.POPULATION],
        lat=projection_row[rc.LATITUDE],
        long=projection_row[rc.LONGITUDE],
        actuals=_generate_actuals(state_latest, state_intervention),
        stateName=projection_row[rc.STATE_FULL_NAME],
        fips=projection_row[rc.FIPS],
        lastUpdatedDate=_format_date(projection_row[rc.LAST_UPDATED]),
        projections=projections,
        timeseries=timeseries,
        actualsTimeseries=_generate_actuals_timeseries(
            actuals_ts.get_records_for_state(state), state_intervention
        ),
    )


def generate_county_timeseries(projection_row, intervention, input_dir):
    state_abbrev = US_STATE_ABBREV[projection_row[rc.STATE_FULL_NAME]]
    fips = projection_row[rc.FIPS]

    raw_dataseries = get_can_projection.get_can_raw_data(
        input_dir, state_abbrev, fips, AggregationLevel.COUNTY, intervention
    )

    testing_df = get_testing_timeseries_by_fips(fips)
    new_df = pd.DataFrame(raw_dataseries).merge(testing_df, on="date", how="left")

    can_dataseries = new_df.to_dict(orient="records")

    timeseries = []
    for data_series in can_dataseries:
        timeseries.append(_generate_county_timeseries_row(data_series))
    if len(timeseries) < 1:
        raise Exception(f"County time series empty for {intervention.name}")

    projections = _generate_api_for_projections(projection_row)
    state_intervention = get_can_projection.get_intervention_for_state(state_abbrev)
    actuals_ts = combined_datasets.build_us_timeseries_with_all_fields()
    actual_latest = combined_datasets.build_us_latest_with_all_fields()
    fips_latest = actual_latest.get_record_for_fips(fips)

    return CovidActNowCountyTimeseries(
        population=fips_latest[CommonFields.POPULATION],
        lat=projection_row[rc.LATITUDE],
        long=projection_row[rc.LONGITUDE],
        actuals=_generate_actuals(fips_latest, state_intervention),
        stateName=projection_row[rc.STATE_FULL_NAME],
        countyName=projection_row[rc.COUNTY],
        fips=projection_row[rc.FIPS],
        lastUpdatedDate=_format_date(projection_row[rc.LAST_UPDATED]),
        projections=projections,
        timeseries=timeseries,
        actualsTimeseries=_generate_actuals_timeseries(
            actuals_ts.get_records_for_fips(fips), state_intervention
        ),
    )


def generate_api_for_state_projection_row(projection_row) -> CovidActNowStateSummary:
    state_abbrev = US_STATE_ABBREV[projection_row[rc.STATE_FULL_NAME]]
    projections = _generate_api_for_projections(projection_row)
    state_intervention = get_can_projection.get_intervention_for_state(state_abbrev)
    state_actuals = combined_datasets.build_us_latest_with_all_fields().get_record_for_state(
        state_abbrev
    )

    state_result = CovidActNowStateSummary(
        population=state_actuals[CommonFields.POPULATION],
        lat=projection_row[rc.LATITUDE],
        long=projection_row[rc.LONGITUDE],
        actuals=_generate_actuals(state_actuals, state_intervention),
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
    fips_actuals = combined_datasets.build_us_latest_with_all_fields().get_record_for_fips(fips)

    county_result = CovidActNowCountySummary(
        population=fips_actuals[CommonFields.POPULATION],
        lat=projection_row[rc.LATITUDE],
        long=projection_row[rc.LONGITUDE],
        actuals=_generate_actuals(fips_actuals, state_intervention),
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
