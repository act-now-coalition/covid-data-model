from datetime import datetime
from api.can_api_definition import (
    CovidActNowCountiesAPI,
    CovidActNowCountySummary,
    _Projections,
    _ResourceUsageProjection,
)
from libs.datasets import results_schema as rc
from libs.constants import NULL_VALUE


def _format_date(input_date):
    if not input_date:
        raise Exception("Can't format a date that doesn't exist")
    if isinstance(input_date, str):
        # note if this is already in iso format it will be grumpy. maybe use dateutil
        datetime_obj = datetime.strptime(input_date, "%m/%d/%Y %H:%M")
        return datetime_obj.isoformat()
    if isinstance(input_date, datetime):
        return input_date.isoformat()
    raise Exception("Invalid date type when converting to api")


def _get_date_or_none(panda_date_or_none):
    """ Projection Null value is a string NULL so if this date value is a string,
     make it none. Otherwise convert to the python datetime. Example
     of this being null is when there is no bed shortfall, the shortfall dates is none """
    if isinstance(panda_date_or_none, str):
        return None
    return _format_date(panda_date_or_none.to_pydatetime())


def _get_or_none(value):
    if isinstance(value, str) and value == NULL_VALUE:
        return None
    else:
        return value


def generate_api_for_projection_row(county_row):
    peak_date = _get_date_or_none(county_row[rc.PEAK_HOSPITALIZATIONS])
    shortage_start_date = _get_date_or_none(county_row[rc.HOSPITAL_SHORTFALL_DATE])
    _hospital_beds = _HospitalBeds(
        peakDate=peak_date,
        shortageStartDate=shortage_start_date,
        peakShortfall=_get_or_none(county_row[rc.PEAK_HOSPITALIZATION_SHORTFALL]),
    )
    _projections = _Projections(hospitalBeds=_hospital_beds)
    county_result = CANPredictionAPIRow(
        stateName=county_row[rc.STATE],
        countyName=county_row[rc.COUNTY],
        fips=county_row[rc.FIPS],
        lastUpdatedDate=_format_date(county_row[rc.LAST_UPDATED]),
        projections=_projections,
    )
    return county_result


def generate_api_for_projection(projection):
    api_results = []

    for index, county_row in projection.iterrows():
        county_result = generate_api_for_projection_row(county_row)
        api_results.append(county_result)
    return CovidActNowCountiesAPI(data=api_results)
