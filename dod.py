import pandas as pd
import numpy as np
import requests
import datetime
import pprint
from us_state_abbrev import us_state_abbrev

# @TODO: Attempt today. If that fails, attempt yesterday.
latest = datetime.date.today() - datetime.timedelta(days=1)

interventions_url = 'https://raw.githubusercontent.com/covid-projections/covid-projections/master/src/assets/data/interventions.json'
interventions = requests.get(interventions_url).json()
interventions_df = pd.DataFrame(
    list(interventions.items()),
    columns=['state', 'intervention']
)
interventions_df.info()

abbrev_df = pd.DataFrame(
    list(us_state_abbrev.items()),
    columns=['state', 'abbreviation']
)
abbrev_df.info()

projections_df = pd.read_csv('projections_03-26-2020.csv')
projections_df.info()

url = 'https://github.com/CSSEGISandData/COVID-19/raw/master/csse_covid_19_data/csse_covid_19_daily_reports/{}.csv'.format(
    latest.strftime("%m-%d-%Y"))
raw_df = pd.read_csv(url)

raw_df.info()

column_mapping = {"Province_State": "Province/State",
                  "Country_Region": "Country/Region",
                  "Last_Update": "Last Update",
                  "Lat": "Latitude",
                  "Long_": "Longitude",
                  "Combined_Key": "Combined Key",
                  "Admin2": "County",
                  "FIPS": "State/County FIPS Code"
                  }

remapped_df = raw_df.rename(columns=column_mapping)
remapped_df.info()


cols = ["Province/State",
        "Country/Region",
        "Last Update",
        "Latitude",
        "Longitude",
        "Confirmed",
        "Recovered",
        "Deaths",
        "Active",
        "County",
        "State/County FIPS Code",
        "Combined Key",
        "Incident Rate",
        "People Tested",
        "Shape"
        ]

final_df = pd.DataFrame(remapped_df, columns=cols)
final_df["Shape"] = "Point"
final_df['Last Update'] = pd.to_datetime(final_df['Last Update'])
final_df['Last Update'] = final_df['Last Update'].dt.strftime(
    '%-m/%-d/%Y %H:%M')

final_df = final_df.fillna("<Null>")

us_only = final_df[(final_df["Country/Region"] == "US")]

pprint.pprint(us_only.head())

us_only.to_csv("results/counties.csv")

states_group = us_only.groupby(['Province/State'])
states_agg = states_group.aggregate({
    'Last Update': 'max',
    'Confirmed': 'sum',
    'Recovered': 'sum',
    'Deaths': 'sum',
    'Active': 'sum',
    'Country/Region': 'first',
    'Latitude': 'first',
    'Longitude': 'first'
    # People tested is currently null
    #'People Tested': 'sum'
}).drop(
    # This value snuck in as a state in the original data
    ['Recovered']
)
states_agg.info()
states_abbrev = states_agg.merge(
    abbrev_df, left_index=True, right_on='state', how='left'
).merge(
    interventions_df, left_on='abbreviation', right_on='state', how='left'
).merge(
    projections_df, left_on='state_y', right_on='State', how='left'
).drop(['abbreviation', 'state_y', 'State'], axis=1)
state_col_remap = {
    'state_x': 'Province/State',
    'Projection1': '4-day Hospitalizations Prediction',
    'Projection2': '8-day Hospitalizations Prediction',
    'intervention': 'Intervention'
}
states_remapped = states_abbrev.rename(columns=state_col_remap)
states_remapped.info()
# TODO: filter out county-specific columns
state_cols = cols + ['Intervention', '4-day Hospitalizations Prediction', '8-day Hospitalizations Prediction']
states_final = pd.DataFrame(states_remapped, columns=state_cols)
states_final['Shape'] = 'Point'
states_final.info()
states_final = states_final.fillna("<Null>")
pprint.pprint(states_final.head())
states_final.to_csv('results/states.csv')
