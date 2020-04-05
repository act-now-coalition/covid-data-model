'''
Constants to use for a build. In a separate file to avoid
auto-importing a dataset when we don't necessarily need to.
'''

from datetime import datetime, timedelta, date


def get_interventions(start_date=datetime.now().date()):
    return [
        None,  # No Intervention
        {  # Flatten the Curve
            start_date: 1.3,
            start_date + timedelta(days=30) : 1.1,
            start_date + timedelta(days=60) : 0.8,
            start_date + timedelta(days=90) : None
        },
        {  # Full Containment
            start_date : 1.3,
            start_date + timedelta(days=7) : 0.3,
            start_date + timedelta(days=30 + 7) : 0.2,
            start_date + timedelta(days=30 + 2*7) : 0.1,
            start_date + timedelta(days=30 + 3*7) : 0.035,
            start_date + timedelta(days=30 + 4*7) : 0
        },
        {  # Social Distancing
            start_date: 1.7,
            start_date + timedelta(days=90) : None
        },
    ]

OUTPUT_DIR = 'results/test'
OUTPUT_DIR_COUNTIES = 'results/county'

# Dict to transform longhand state names to abbreviations
US_STATE_ABBREV =  {
    'Alabama': 'AL',
    'Alaska': 'AK',
    'American Samoa': 'AS',
    'Arizona': 'AZ',
    'Arkansas': 'AR',
    'California': 'CA',
    'Colorado': 'CO',
    'Connecticut': 'CT',
    'Delaware': 'DE',
    'District of Columbia': 'DC',
    'Florida': 'FL',
    'Georgia': 'GA',
    'Guam': 'GU',
    'Hawaii': 'HI',
    'Idaho': 'ID',
    'Illinois': 'IL',
    'Indiana': 'IN',
    'Iowa': 'IA',
    'Kansas': 'KS',
    'Kentucky': 'KY',
    'Louisiana': 'LA',
    'Maine': 'ME',
    'Maryland': 'MD',
    'Massachusetts': 'MA',
    'Michigan': 'MI',
    'Minnesota': 'MN',
    'Mississippi': 'MS',
    'Missouri': 'MO',
    'Montana': 'MT',
    'Nebraska': 'NE',
    'Nevada': 'NV',
    'New Hampshire': 'NH',
    'New Jersey': 'NJ',
    'New Mexico': 'NM',
    'New York': 'NY',
    'North Carolina': 'NC',
    'North Dakota': 'ND',
    'Northern Mariana Islands': 'MP',
    'Ohio': 'OH',
    'Oklahoma': 'OK',
    'Oregon': 'OR',
    'Pennsylvania': 'PA',
    'Puerto Rico': 'PR',
    'Rhode Island': 'RI',
    'South Carolina': 'SC',
    'South Dakota': 'SD',
    'Tennessee': 'TN',
    'Texas': 'TX',
    'Utah': 'UT',
    'Vermont': 'VT',
    'Virgin Islands': 'VI',
    'Virginia': 'VA',
    'Washington': 'WA',
    'West Virginia': 'WV',
    'Wisconsin': 'WI',
    'Wyoming': 'WY'
}
