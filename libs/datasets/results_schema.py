RESULT_DATA_COLUMNS_SHARED = [
    'Province/State',
    "Country/Region",
    "Last Update",
    "Latitude",
    "Longitude",
    "State/County FIPS Code",
    'State Intervention',
    '16d-HSPTLZD',
    '32d-HSPTLZD',
    '16d-LACKBEDS',
    '32d-LACKBEDS',
    'MEAN-HOSP',
    'MEAN-DEATHS',
    'PEAK-HOSP',
    'PEAK-DEATHS',
    "Current Deaths",
    "Current Confirmed",
    # "Current Recovered", # these are always zero
    # "Current Active", # these are always zero
    "Combined Key",
    "County"
]

RESULT_DATA_COLUMNS_STATES = RESULT_DATA_COLUMNS_SHARED + []
RESULT_DATA_COLUMNS_COUNTIES = RESULT_DATA_COLUMNS_SHARED + []

EXPECTED_MISSING_STATES = set([
    'Northern Mariana Islands', 'American Samoa', 'Virgin Islands', 'Puerto Rico', 'Guam'
])

EXPECTED_MISSING_STATES_FROM_COUNTES = set([
    'District of Columbia'
])