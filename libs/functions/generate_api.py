from api.can_predictions import CANPredictionAPIRow, CANPredictionAPI, _Projections, _HospitalBeds
from libs.datasets import results_schema as rc

def _get_date_or_none(panda_date_or_none):
    """ Projection Null value is a string NULL so if this date value is a string,
     make it none. Otherwise convert to the python datetime. Example 
     of this being null is when there is no bed shortfall, the shortfall dates is none """
    if isinstance(panda_date_or_none, str):
        return None
    return panda_date_or_none.to_pydatetime()

def generate_api_for_projection(projection): 
    api_results = []

    for index, county_row in projection.iterrows(): 
        peak_date = _get_date_or_none(county_row[rc.PEAK_HOSPITALIZATIONS])          
        shortage_start_date = _get_date_or_none(county_row[rc.HOSPITAL_SHORTFALL_DATE])
        _hospital_beds = _HospitalBeds(
            peakDate=peak_date, 
            shortageStartDate=shortage_start_date, 
            peakShortfall=county_row[rc.PEAK_HOSPITALIZATION_SHORTFALL])
        _projections = _Projections(hospitalBeds=_hospital_beds, aggregateDeaths=county_row[rc.MEAN_DEATHS])
        county_result = CANPredictionAPIRow(stateName=county_row[rc.STATE], countyName=county_row[rc.COUNTY], fips=county_row[rc.FIPS], lastUpdatedDate=county_row[rc.LAST_UPDATED], projections=_projections)
        api_results.append(county_result)
    return CANPredictionAPI(data=api_results)