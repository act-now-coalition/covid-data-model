OUTPUT_COLUMN_REMAP_TO_RESULT_DATA = {
    'fips': "State/County FIPS Code",
    'state': 'Province/State',
    'country': 'Country/Region',
    'county': 'County',
    'date': 'Last Update',
    'intervention': 'State Intervention',
    '16-day_Hospitalization_Prediction': '16d-HSPTLZD',
    '32-day_Hospitalization_Prediction': '32d-HSPTLZD',
    '16-day_Beds_Shortfall': '16d-LACKBEDS',
    '32-day_Beds_Shortfall': '32d-LACKBEDS',
    "Mean Hospitalizations": 'MEAN-HOSP',
    "Mean Deaths": 'MEAN-DEATHS',
    "Peak Hospitalizations On": 'PEAK-HOSP',
    "Mean Deaths On": 'PEAK-DEATHS',
    "deaths": "Current Deaths",
    "cases": "Current Confirmed",
    "recovered": "Current Recovered",
}


CALCULATED_PROJECTION_HEADERS_SHARED = [
    '16-day_Hospitalization_Prediction',
    '32-day_Hospitalization_Prediction',
    '16-day_Beds_Shortfall',
    '32-day_Beds_Shortfall',
    "Mean Hospitalizations",
    "Mean Deaths",
    "Peak Hospitalizations On",
    "Mean Deaths On",
    "Hospital Shortfall Date",
    "Peak Hospitlizations Shortfall",
    "Beds at Peak Hospitilization Date",
]

CALCULATED_PROJECTION_HEADERS_STATES = ['State'] + CALCULATED_PROJECTION_HEADERS_SHARED
CALCULATED_PROJECTION_HEADERS_COUNTIES = ['State', 'FIPS'] + CALCULATED_PROJECTION_HEADERS_SHARED
