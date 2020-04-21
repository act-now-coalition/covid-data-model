import pandas as pd
import datetime
import os.path
import simplejson

from libs.us_state_abbrev import US_STATE_ABBREV
from libs.datasets import FIPSPopulation
from libs.datasets.can_model_output_schema import (
    CAN_MODEL_OUTPUT_SCHEMA,
    CAN_MODEL_OUTPUT_SCHEMA_EXCLUDED_COLUMNS,
)
from libs.datasets.projections_schema import (
    CALCULATED_PROJECTION_HEADERS_STATES,
    CALCULATED_PROJECTION_HEADERS_COUNTIES,
)
from libs.enums import Intervention
from libs.constants import NULL_VALUE

def _calc_short_fall(x):
    return abs(x.beds - x.all_hospitalized) if x.all_hospitalized > x.beds else 0


def _get_hospitals_and_shortfalls(df, date_out):
    first_record_after_date = df[(df.date > date_out)].iloc[0]
    hospital_short_fall_columns = ["all_hospitalized", "short_fall"]
    return tuple(first_record_after_date[hospital_short_fall_columns].values)

def _beds_after_given_date(df, date_out):
    if df[(df.date > date_out)].empty:
        return df["beds"].max()
    first_record_after_date = df[(df.date > date_out)].iloc[0]
    bed_columns = ["beds"]
    return first_record_after_date[bed_columns].values[0]

def _read_json_as_df(path):
    # TODO: read this from a dataset class
    df = pd.DataFrame.from_records(
        simplejson.load(open(path, "r")),
        columns=CAN_MODEL_OUTPUT_SCHEMA,
        exclude=CAN_MODEL_OUTPUT_SCHEMA_EXCLUDED_COLUMNS,
    )

    df["date"] = pd.to_datetime(df.date, format='%m/%d/%y')
    df["all_hospitalized"] = df["all_hospitalized"].astype("int")
    df["beds"] = df["beds"].astype("int")
    df["dead"] = df["dead"].astype("int")
    df["population"] = df["population"].astype("int")
    return df


def _calculate_projection_data(state, fips, file_path):
    """
    Given a file path, return the calculations we perform for that file.
    Note in the future maybe return a data type to keep type clarity
    """
    # get 16 and 32 days out from now
    today = datetime.datetime.now()
    sixteen_days = today + datetime.timedelta(days=16)
    thirty_two_days = today + datetime.timedelta(days=32)

    record = {}

    if os.path.exists(file_path):
        df = _read_json_as_df(file_path)
        df["short_fall"] = df.apply(_calc_short_fall, axis=1)

        hosp_16_days, short_fall_16_days = _get_hospitals_and_shortfalls(
            df, sixteen_days
        )
        hosp_32_days, short_fall_32_days = _get_hospitals_and_shortfalls(
            df, thirty_two_days
        )

        df["new_deaths"] = df.dead - df.dead.shift(1)
        hospitals_shortfall_date = NULL_VALUE
        if not df[(df['short_fall']>0)].empty:
            hospitals_shortfall_date =  df[(df['short_fall']>0)].iloc[0].date
        mean_hospitalizations = df.all_hospitalized.mean().round(0)
        mean_deaths = df.new_deaths.mean()

        peak_hospitalizations_date = df.iloc[df.all_hospitalized.idxmax()].date
        beds_at_peak_hospitalization_date = _beds_after_given_date(df, peak_hospitalizations_date)
        peak_hospitalizations_short_falls = df.iloc[df.all_hospitalized.idxmax()].short_fall
        peak_deaths_date = df.iloc[df.new_deaths.idxmax()].date
        population = df.iloc[0].population

        record['State'] = state
        record['FIPS'] = fips
        record['16-day_Hospitalization_Prediction'] = hosp_16_days
        record['32-day_Hospitalization_Prediction'] = hosp_32_days
        record['16-day_Beds_Shortfall'] = short_fall_16_days
        record['32-day_Beds_Shortfall'] = short_fall_32_days
        record['Mean Hospitalizations'] = mean_hospitalizations
        record['Mean Deaths'] = mean_deaths
        record['Peak Hospitalizations On'] = peak_hospitalizations_date
        record['Peak Deaths On'] = peak_deaths_date
        record['Hospital Shortfall Date'] = hospitals_shortfall_date
        record['Peak Hospitlizations Shortfall'] = peak_hospitalizations_short_falls
        record['Beds at Peak Hospitilization Date'] = beds_at_peak_hospitalization_date
        record['Population'] = population

        return pd.Series(record)
    return pd.Series(record)

def _get_intervention_type(intervention_type, state, state_interventions_df):
    if intervention_type == Intervention.SELECTED_MITIGATION.value:
        state_intervention_results = state_interventions_df.loc[state_interventions_df["state"] == state]["intervention"]
        if not state_intervention_results.empty:
            intervention_string = state_intervention_results.values[0]
            return Intervention.from_str(intervention_string).value
    return intervention_type

def get_state_projections_df(input_dir, initial_intervention_type, state_interventions_df):
    """
    for each state in our data look at the results we generated via run.py
    to create the projections
    """

    # save results in a list of lists, converted to df later
    results = []
    for state in list(US_STATE_ABBREV.values()):
        intervention_type = _get_intervention_type(initial_intervention_type, state, state_interventions_df)
        file_name = f"{state}.{intervention_type}.json"
        path = os.path.join(input_dir, file_name)
        # if the file exists in that directory then process
        projection_data = _calculate_projection_data(path)
        if projection_data:
            results.append([state] + projection_data)
    return pd.DataFrame(results, columns=CALCULATED_PROJECTION_HEADERS_STATES)


def get_file_path(input_dir, state, fips, intervention_type):
    file_name = f"{state}.{fips}.{intervention_type}.json"
    return os.path.join(input_dir, file_name)

from tqdm import tqdm
tqdm.pandas()


def get_county_projections_df(input_dir, initial_intervention_type, state_interventions_df):
    """
    for each state in our data look at the results we generated via run.py
    to create the projections
    """
    fips_pd = FIPSPopulation.local().data  # to get the state, county & fips

    fdf = fips_pd[['state' ,'fips']]
    fdf.loc[:,'intervention_type'] = fdf.state.apply(
        lambda x: _get_intervention_type(initial_intervention_type, x, state_interventions_df)
        )
    fdf.loc[:, 'path'] = fdf.apply(
        lambda x: get_file_path(input_dir, x.state, x.fips, x.intervention_type),
        axis=1).values
    ndf = fdf.progress_apply(
        lambda x:_calculate_projection_data(x.state, x.fips, x.path), axis=1)

    missing = ndf.isnull().sum()['State']

    if (missing > 2000):
        raise Exception(f"Missing a majority of counties from input_dir: {input_dir}")
    print(f"Models missing for {missing} counties")

    #ndf = pd.DataFrame(results, columns=CALCULATED_PROJECTION_HEADERS_COUNTIES)
    return ndf
