import pandas as pd
import numpy as np
import requests
import datetime
import pprint
import shapefile
from urllib.parse import urlparse

from .CovidDatasets import get_public_data_base_url
from .us_state_abbrev import us_state_abbrev, us_fips

# @TODO: Attempt today. If that fails, attempt yesterday.
latest = datetime.date.today() - datetime.timedelta(days=1)

NULL_VALUE = "<Null>"

def _file_uri_to_path(uri: str) -> str:
    return urlparse(uri).path

def get_interventions_df():
    interventions_url = 'https://raw.githubusercontent.com/covid-projections/covid-projections/master/src/assets/data/interventions.json'
    interventions = requests.get(interventions_url).json()
    return pd.DataFrame(
        list(interventions.items()),
        columns=['state', 'intervention']
    )

def get_abbrev_df():
    return pd.DataFrame(
        list(us_state_abbrev.items()),
        columns=['state', 'abbreviation']
    )

def get_projections_3_26_df():
    return pd.read_csv('projections_03-26-2020.csv')

def get_projections_df():
    return pd.read_csv('projections_3_31_2020.csv')

output_cols = ["Province/State",
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

new_cols = ['16-day_Hospitalization_Prediction',
    '32-day_Hospitalization_Prediction',
    '16-day_Beds_Shortfall',
    '32-day_Beds_Shortfall']

county_replace_with_null = {
    "Unassigned": NULL_VALUE
}

def get_usa_by_county_df():
    url = '{}/data/cases-jhu/csse_covid_19_daily_reports/{}.csv'.format(
        get_public_data_base_url(), latest.strftime("%m-%d-%Y"))
    raw_df = pd.read_csv(url)

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

    # USA only
    us_df = remapped_df[(remapped_df["Country/Region"] == "US")]

    final_df = pd.DataFrame(us_df, columns=output_cols)
    final_df["Shape"] = "Point"
    final_df['Last Update'] = pd.to_datetime(final_df['Last Update'])
    final_df['Last Update'] = final_df['Last Update'].dt.strftime(
        '%-m/%-d/%Y %H:%M')

    final_df['County'] = final_df['County'].replace(county_replace_with_null)
    final_df['Combined Key'] = final_df['Combined Key'].str.replace('Unassigned, ','')
    final_df = final_df.fillna(NULL_VALUE)
    # handle serializing FIPS without trailing 0
    final_df['State/County FIPS Code'] = final_df['State/County FIPS Code'].astype(str).str.replace('\.0','')

    final_df.index.name = 'OBJECTID'
    # assert unique key test
    assert final_df['Combined Key'].value_counts().max() == 1

    return final_df


def get_usa_by_states_df():

    us_only = get_usa_by_county_df()
    abbrev_df = get_abbrev_df()
    interventions_df = get_interventions_df()
    projections_df = get_projections_df()
    print(projections_df)

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
    })

    states_abbrev = states_agg.merge(
        abbrev_df, left_index=True, right_on='state', how='left'
    ).merge(
        # inner merge to filter to only the 50 states
        interventions_df, left_on='abbreviation', right_on='state', how='inner'
    ).merge(
        projections_df, left_on='state_y', right_on='State', how='left'
    ).drop(['abbreviation', 'state_y', 'State'], axis=1)

    state_col_remap = {
        'state_x': 'Province/State',
        'Projection1': '4-day Hospitalizations Prediction',
        'Projection2': '8-day Hospitalizations Prediction',
        'intervention': 'Intervention'
    }

    # 'Projection1': '16-day_Hopitalization_Prediction',
    # 'Projection2': '32-day_Hospitalization_Prediction',
    # 'Projection1': '16-day_Beds_Shortfall',
    # 'Projection2': '32-day_Beds_Shortwall',


    states_remapped = states_abbrev.rename(columns=state_col_remap)

    # TODO: filter out county-specific columns
    state_cols = output_cols + ['Intervention', '4-day Hospitalizations Prediction', '8-day Hospitalizations Prediction']
    state_cols += new_cols

    states_final = pd.DataFrame(states_remapped, columns=state_cols)
    states_final['Shape'] = 'Point'
    states_final = states_final.fillna(NULL_VALUE)
    states_final['Combined Key'] = states_final['Province/State']
    states_final['State/County FIPS Code'] = states_final['Province/State'].map(us_fips)

    # temporarily nullify predictions
    states_final['8-day Hospitalizations Prediction'] = NULL_VALUE
    states_final['4-day Hospitalizations Prediction'] = NULL_VALUE

    states_final.index.name = 'OBJECTID'
    # assert unique key test
    assert states_final['Combined Key'].value_counts().max() == 1

    return states_final

# note it's unclear to me if 'Incident Rate' is a number, float, etc
def join_and_output_shapefile(df, shp_reader, pivot_shp_field, pivot_df_column, shp_writer):
    fields = ['Confirmed', 'Recovered', 'Deaths', 'Active', 'Incident Rate', 'People Tested', 'Intervention', '4-day Hospitalizations Prediction', '8-day Hospitalizations Prediction'] + new_cols
    fields = [field for field in fields if field in df.columns]

    shp_writer.fields = shp_reader.fields # Preserve fields that come from the census

    for field_name in fields:
        if field_name == 'Intervention': # Intervention is currently our only non-integer field
            shp_writer.field(field_name, 'C', size=32)
        else:
            shp_writer.field(field_name, 'N', size=14)

    for shapeRecord in shp_reader.shapeRecords():
        try:
            row = df[df[pivot_df_column] == shapeRecord.record[pivot_shp_field]].iloc[0]
        except:
            continue

        new_record = shapeRecord.record.as_dict()
        for field_name in fields:
            # random bad data seems to come back as this weird string, not too sure about this
            new_record[field_name] = None if row[field_name] == '<Null>' else row[field_name]
        shp_writer.shape(shapeRecord.shape)
        shp_writer.record(**new_record)

    shp_writer.close()


def get_usa_state_shapefile(shp, shx, dbf):
    shp_writer = shapefile.Writer(shp=shp, shx=shx, dbf=dbf)
    public_data_url = get_public_data_base_url()
    public_data_path = _file_uri_to_path(public_data_url)
    join_and_output_shapefile(get_usa_by_states_df(),
        shapefile.Reader(f'{public_data_path}/data/shapefiles-uscensus/tl_2019_us_state'),
        'STATEFP', 'State/County FIPS Code', shp_writer)

def get_usa_county_shapefile(shp, shx, dbf):
    df = get_usa_by_county_df()
    shp_writer = shapefile.Writer(shp=shp, shx=shx, dbf=dbf)
    # ironically we have to re-pad the dataframe column to easily match GEOID in the shapefile
    df['State/County FIPS Code'] = df['State/County FIPS Code'].astype(str).str.rjust(5, '0')

    public_data_url = get_public_data_base_url()
    public_data_path = _file_uri_to_path(public_data_url)
    join_and_output_shapefile(df,
        shapefile.Reader(f'{public_data_path}/data/shapefiles-uscensus/tl_2019_us_county'),
        'GEOID', 'State/County FIPS Code', shp_writer)

# us_only = get_usa_by_county_df()
# us_only.to_csv("results/counties.csv")

# states_final = get_usa_by_states_df()
# states_final.to_csv('results/states.csv')
