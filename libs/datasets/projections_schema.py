from libs.datasets import CommonFields

OUTPUT_COLUMN_REMAP_TO_RESULT_DATA = {
    'intervention': 'State Intervention',
    '16-day_Hospitalization_Prediction': '16d-HSPTLZD',
    '32-day_Hospitalization_Prediction': '32d-HSPTLZD',
    '16-day_Beds_Shortfall': '16d-LACKBEDS',
    '32-day_Beds_Shortfall': '32d-LACKBEDS',
    "Mean Hospitalizations": 'MEAN-HOSP',
    "Mean Deaths": 'MEAN-DEATHS',
    "Peak Hospitalizations On": 'PEAK-HOSP',
    "Peak Deaths On": 'PEAK-DEATHS',
    "Deaths": "Current Deaths",
    "Confirmed": "Current Confirmed",
    "Recovered": "Current Recovered",
    "Active": "Current Active",
    "Beds at Peak Hospitilization Date": "Peak Bed Capacity",
    "Population": "Population",
    "Rt": "Rt",
    "Rt_ci90": "Rt_ci90"
}

CALCULATED_PROJECTION_HEADERS_SHARED = [
    '16-day_Hospitalization_Prediction',
    '32-day_Hospitalization_Prediction',
    '16-day_Beds_Shortfall',
    '32-day_Beds_Shortfall',
    "Mean Hospitalizations",
    "Mean Deaths",
    "Peak Hospitalizations On",
    "Peak Deaths On",
    "Hospital Shortfall Date",
    "Peak Hospitlizations Shortfall",
    "Beds at Peak Hospitilization Date",
    "Population",
    "Rt",
    "Rt_ci90"
]

CALCULATED_PROJECTION_HEADERS_STATES = [CommonFields.STATE_FULL_NAME] + CALCULATED_PROJECTION_HEADERS_SHARED
CALCULATED_PROJECTION_HEADERS_COUNTIES = [CommonFields.STATE_FULL_NAME, CommonFields.FIPS] + CALCULATED_PROJECTION_HEADERS_SHARED
