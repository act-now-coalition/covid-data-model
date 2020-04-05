import pandas as pd
import numpy as np
import requests
import datetime
import os.path
import pprint
import shapefile
import simplejson
import statistics
import math, sys

from urllib.parse import urlparse
from collections import defaultdict

from libs.constants import NO_INTERVENTION, DATE, ALL_HOSPITALIZED, ALL_INFECTED, DEAD, BEDS
from libs.build_params import OUTPUT_DIR, OUTPUT_DIR_COUNTIES
from libs.CovidDatasets import get_public_data_base_url
from libs.us_state_abbrev import us_state_abbrev, us_fips
from libs.datasets import FIPSPopulation

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

all_cols = [
    "day_num",
    # ^ for index column
    "date",
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "all_hospitalized",
    "all_infected",
    "dead",
    "beds",
    "i",
    "j",
    "k",
    "l",
    "population",
    "m",
    "n",
]

exclude_cols = ["a","b","c","d","e","f","g","i","j","k","l","m","n"]

def read_json_as_df(path):
    df = pd.DataFrame.from_records(simplejson.load(open(path,'r')),
    columns=all_cols, exclude=exclude_cols)

    df['date'] = pd.to_datetime(df.date)
    df['all_hospitalized'] = df['all_hospitalized'].astype('int')
    df['beds'] = df['beds'].astype('int')
    df['dead'] = df['dead'].astype('int')
    return df

def calc_short_fall(x):
    return abs(x.beds - x.all_hospitalized) if x.all_hospitalized > x.beds else 0

def get_hospitals_and_shortfalls(df, date_out):
    first_record_after_date = df[(df.date > date_out)].iloc[0]
    output_cols = ['all_hospitalized', 'short_fall']
    return tuple(first_record_after_date[output_cols].values)

def get_projections_df():
    # for each state in our data look at the results we generated via run.py
    # to create the projections
    intervention_type = NO_INTERVENTION # None, as requested

    # get 16 and 32 days out from now
    today = datetime.datetime.now()
    sixteen_days = today + datetime.timedelta(days=16)
    thirty_two_days = today + datetime.timedelta(days=32)

    #save results in a list of lists, converted to df later
    results = []

    for state in list(us_state_abbrev.values()):
        file_name = f'{state}.{intervention_type}.json'
        path = os.path.join(OUTPUT_DIR, file_name)

        # if the file exists in that directory then process
        if os.path.exists(path):
            df = read_json_as_df(path)
            df['short_fall'] = df.apply(calc_short_fall, axis=1)

            hosp_16_days, short_fall_16_days = get_hospitals_and_shortfalls(df, sixteen_days)
            hosp_32_days, short_fall_32_days = get_hospitals_and_shortfalls(df, thirty_two_days)

            results.append([state, hosp_16_days, hosp_32_days, short_fall_16_days, short_fall_32_days])

    headers = [
        'State',
        '16-day_Hospitalization_Prediction',
        '32-day_Hospitalization_Prediction',
        '16-day_Beds_Shortfall','32-day_Beds_Shortfall'
    ] # used for pandas
    return pd.DataFrame(results, columns=headers)

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
    # Incident rate and people tested do not seem to be available yet
    # "Incident Rate",
    # "People Tested",
]

county_output_cols = ["Province/State",
    "Country/Region",
    "Last Update",
    "Latitude",
    "Longitude",
    "Current Confirmed",
    # "Current Recovered",
    # "Current Deaths",
    "Current Active",
    "County",
    "State/County FIPS Code",
    "Combined Key",
    # Incident rate and people tested do not seem to be available yet
    # "Incident Rate",
    # "People Tested",
]

county_replace_with_null = {
    "Unassigned": NULL_VALUE
}


def get_county_projections():
    # for each state in our data look at the results we generated via run.py
    # to create the projections
    fips_pd = FIPSPopulation.local().data # to get the state, county & fips
    intervention_type = NO_INTERVENTION # None, as requested

    # get 16 and 32 days out from now
    today = datetime.datetime.now()
    sixteen_days = today + datetime.timedelta(days=16)
    thirty_two_days = today + datetime.timedelta(days=32)

    #save results in a list of lists, converted to df later
    results = []

    # get the state and fips so we can get the files
    missing = 0
    for index, fips_row in fips_pd.iterrows():
        state = fips_row['state']
        fips = fips_row['fips']
        file_name = f"{state}.{fips}.{intervention_type}.json"
        path = os.path.join(OUTPUT_DIR_COUNTIES, file_name)
        # if the file exists in that directory then process
        if os.path.exists(path):
            df = read_json_as_df(path)
            df['short_fall'] = df.apply(calc_short_fall, axis=1)

            hosp_16_days, short_fall_16_days = get_hospitals_and_shortfalls(df, sixteen_days)
            hosp_32_days, short_fall_32_days = get_hospitals_and_shortfalls(df, thirty_two_days)

            #hospitalizations = [int(row[9]) for row in projection]
            #deaths = [int(row[11]) for row in projection]
            df['new_deaths'] = df.dead - df.dead.shift(1)

            mean_hospitalizations = df.all_hospitalized.mean().round(0)
            # mean_hospitalizations = math.floor(statistics.mean(hospitalizations))
            mean_deaths = df.new_deaths.mean()

            peak_hospitalizations_date = df.iloc[df.all_hospitalized.idxmax()].date
            peak_deaths_date = df.iloc[df.new_deaths.idxmax()].date

            results.append([state, fips, hosp_16_days, hosp_32_days, short_fall_16_days, short_fall_32_days,
                    mean_hospitalizations, mean_deaths, peak_hospitalizations_date, peak_deaths_date])
        else:
            missing = missing + 1
    print(f'Models missing for {missing} county')

    headers = [
        'State',
        'FIPS',
        '16-day_Hospitalization_Prediction',
        '32-day_Hospitalization_Prediction',
        '16-day_Beds_Shortfall',
        '32-day_Beds_Shortfall',
        "Mean Hospitalizations",
        "Mean Deaths",
        "Peak Hospitalizations On",
        "Mean Deaths On",
    ]
    ndf = pd.DataFrame(results, columns=headers)
    return ndf


def get_usa_by_county_with_projection_df():
    us_only = get_usa_by_county_df()
    fips_df = FIPSPopulation.local().data # used to get interventions
    interventions_df = get_interventions_df() # used to say what state has what interventions
    projections_df = get_county_projections()

    counties_decorated = us_only.merge(
        projections_df, left_on='State/County FIPS Code', right_on='FIPS', how='inner'
    ).merge(
        fips_df[['state', 'fips']], left_on='FIPS', right_on='fips', how='inner'
    ).merge(
        interventions_df, left_on='state', right_on='state', how = 'inner'
    )

    state_col_remap = {
        'state_x': 'Province/State',
        'intervention': 'State Intervention',
        '16-day_Hospitalization_Prediction': '16d-HSPTLZD',
        '32-day_Hospitalization_Prediction': '32d-HSPTLZD',
        '16-day_Beds_Shortfall': '16d-LACKBEDS',
        '32-day_Beds_Shortfall': '32d-LACKBEDS',
        "Mean Hospitalizations": 'MEAN-HOSP',
        "Mean Deaths": 'MEAN-DEATHS',
        "Peak Hospitalizations On": 'PEAK-HOSP',
        "Mean Deaths On": 'PEAK-DEATHS',
        "Deaths": "Current Deaths",
        "Confirmed": "Current Confirmed",
        "Recovered": "Current Recovered",
        "Active": "Current Active",
    }

    counties_remapped = counties_decorated.rename(columns=state_col_remap)

    new_cols = list(set(county_output_cols + list(state_col_remap.values())))
    counties = pd.DataFrame(counties_remapped, columns=new_cols)
    counties = counties.fillna(NULL_VALUE)
    counties.index.name = 'OBJECTID'
    # assert unique key test
    assert counties['Combined Key'].value_counts().max() == 1
    return counties

def get_usa_by_county_df():
    url = '{}/data/cases-jhu/csse_covid_19_daily_reports/{}.csv'.format(
        get_public_data_base_url(), latest.strftime("%m-%d-%Y"))
    raw_df = pd.read_csv(url, dtype={"FIPS": str})
    raw_df['FIPS'] = raw_df['FIPS'].astype(str).str.zfill(5)

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
    final_df['Last Update'] = pd.to_datetime(final_df['Last Update'])
    final_df['Last Update'] = final_df['Last Update'].dt.strftime(
        '%-m/%-d/%Y %H:%M')

    final_df['County'] = final_df['County'].replace(county_replace_with_null)
    final_df['Combined Key'] = final_df['Combined Key'].str.replace('Unassigned, ','')
    final_df = final_df.fillna(NULL_VALUE)

    final_df.index.name = 'OBJECTID'
    # assert unique key test
    assert final_df['Combined Key'].value_counts().max() == 1

    return final_df


def get_usa_by_states_df():

    us_only = get_usa_by_county_df()
    abbrev_df = get_abbrev_df()
    interventions_df = get_interventions_df()
    projections_df = get_projections_df()

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

    # basically the states_agg has full state names, the interventions have abbreviation so we need these to be merged
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
        'intervention': 'Intervention',
        '16-day_Hospitalization_Prediction': '16d-HSPTLZD',
        '32-day_Hospitalization_Prediction': '32d-HSPTLZD',
        '16-day_Beds_Shortfall': '16d-LACKBEDS',
        '32-day_Beds_Shortfall': '32d-LACKBEDS'
    }

    states_remapped = states_abbrev.rename(columns=state_col_remap)

    new_cols = list(set(output_cols + list(state_col_remap.values())))
    states_final = pd.DataFrame(states_remapped, columns=new_cols)
    states_final = states_final.fillna(NULL_VALUE)
    states_final['Combined Key'] = states_final['Province/State']
    states_final['State/County FIPS Code'] = states_final['Province/State'].map(us_fips)

    # Missing 4d/8d numberse from model?
    states_final['4d-HSPTLZD'] = NULL_VALUE
    states_final['8d-HSPTLZD'] = NULL_VALUE

    states_final.index.name = 'OBJECTID'
    # assert unique key test
    assert states_final['Combined Key'].value_counts().max() == 1

    return states_final


def join_and_output_shapefile(df, shp_reader, pivot_shp_field, pivot_df_column, shp_writer):
    blacklisted_fields = ['OBJECTID', 'Province/State', 'Country/Region', 'Last Update',
        'Latitude', 'Longitude', 'County',  'State/County FIPS Code',
        'Combined Key', 'Current Recovered', 'Current Active', "Recovered", "Active"]
    non_integer_fields = ['Intervention', 'State Intervention', 'PEAK-HOSP', 'PEAK-DEATHS']

    fields = [field for field in df.columns if field not in blacklisted_fields]

    shp_writer.fields = shp_reader.fields # Preserve fields that come from the census

    for field_name in fields:
        if field_name in non_integer_fields:
            shp_writer.field(field_name, 'C', size=32)
        else:
            shp_writer.field(field_name, 'N', size=14)

    shapeRecords = shp_reader.shapeRecords()
    # if you are using a local copy of the data, LFS truncates the records
    assert len(shapeRecords) >= 50

    # Just adding some understanding of errors
    failed_dictionary = {}

    for shapeRecord in shapeRecords:
        try:
            # Gets the row of the dataframe that matches the FIPS codes for a state/county
            row = df[df[pivot_df_column] == shapeRecord.record[pivot_shp_field]].iloc[0]
        except Exception as e:
            state_fips = shapeRecord.record[pivot_shp_field][:2] # state fips is the first two chars of the state/county fips
            failed_dictionary.setdefault(state_fips, []).append(shapeRecord.record[pivot_shp_field])
            continue

        # create record data for all the fields create a shape record
        new_record = shapeRecord.record.as_dict()
        for field_name in fields:
            new_record[field_name] = None if row[field_name] == NULL_VALUE else row[field_name]
        shp_writer.shape(shapeRecord.shape)
        shp_writer.record(**new_record)

    # Different errors in each state, note this includes the territories as well
    print([(state, len(failed_dictionary[state])) for state in failed_dictionary])
    shp_writer.close()


def get_usa_state_shapefile(shp, shx, dbf):
    shp_writer = shapefile.Writer(shp=shp, shx=shx, dbf=dbf)
    public_data_url = get_public_data_base_url()
    public_data_path = _file_uri_to_path(public_data_url)
    join_and_output_shapefile(get_usa_by_states_df(),
        shapefile.Reader(f'{public_data_path}/data/shapefiles-uscensus/tl_2019_us_state'),
        'STATEFP', 'State/County FIPS Code', shp_writer)

def get_usa_county_shapefile(shp, shx, dbf):
    shp_writer = shapefile.Writer(shp=shp, shx=shx, dbf=dbf)
    public_data_url = get_public_data_base_url()
    public_data_path = _file_uri_to_path(public_data_url)

    join_and_output_shapefile(get_usa_by_county_with_projection_df(),
        shapefile.Reader(f'{public_data_path}/data/shapefiles-uscensus/tl_2019_us_county'),
        'GEOID', 'State/County FIPS Code', shp_writer)

# us_only = get_usa_by_county_df()
# us_only.to_csv("results/counties.csv")

# states_final = get_usa_by_states_df()
# states_final.to_csv('results/states.csv')
