import pandas as pd
import datetime
import os.path
import simplejson

from libs.us_state_abbrev import us_state_abbrev
from libs.datasets import FIPSPopulation
from libs.datasets.can_model_output_schema import (
    CAN_MODEL_OUTPUT_SCHEMA,
    CAN_MODEL_OUTPUT_SCHEMA_EXCLUDED_COLUMNS,
)
from libs.datasets.projections_schema import (
    CALCULATED_PROJECTION_HEADERS_STATES,
    CALCULATED_PROJECTION_HEADERS_COUNTIES,
)


def _calc_short_fall(x):
    return abs(x.beds - x.all_hospitalized) if x.all_hospitalized > x.beds else 0


def _get_hospitals_and_shortfalls(df, date_out):
    first_record_after_date = df[(df.date > date_out)].iloc[0]
    hospital_short_fall_columns = ["all_hospitalized", "short_fall"]
    return tuple(first_record_after_date[hospital_short_fall_columns].values)


def _read_json_as_df(path):
    # TODO: read this from a dataset class
    df = pd.DataFrame.from_records(
        simplejson.load(open(path, "r")),
        columns=CAN_MODEL_OUTPUT_SCHEMA,
        exclude=CAN_MODEL_OUTPUT_SCHEMA_EXCLUDED_COLUMNS,
    )

    df["date"] = pd.to_datetime(df.date)
    df["all_hospitalized"] = df["all_hospitalized"].astype("int")
    df["beds"] = df["beds"].astype("int")
    df["dead"] = df["dead"].astype("int")
    return df


def _calculate_projection_data(file_path):
    """
    Given a file path, return the calculations we perform for that file.
    Note in the future maybe return a data type to keep type clarity
    """
    # get 16 and 32 days out from now
    today = datetime.datetime.now()
    sixteen_days = today + datetime.timedelta(days=16)
    thirty_two_days = today + datetime.timedelta(days=32)

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

        mean_hospitalizations = df.all_hospitalized.mean().round(0)
        mean_deaths = df.new_deaths.mean()

        peak_hospitalizations_date = df.iloc[df.all_hospitalized.idxmax()].date
        peak_deaths_date = df.iloc[df.new_deaths.idxmax()].date

        return [
            hosp_16_days,
            hosp_32_days,
            short_fall_16_days,
            short_fall_32_days,
            mean_hospitalizations,
            mean_deaths,
            peak_hospitalizations_date,
            peak_deaths_date,
        ]
    return None


def get_state_projections_df(input_dir, intervention_type):
    """
    for each state in our data look at the results we generated via run.py
    to create the projections
    """

    # save results in a list of lists, converted to df later
    results = []

    for state in list(us_state_abbrev.values()):
        file_name = f"{state}.{intervention_type}.json"
        path = os.path.join(input_dir, "state", file_name)

        # if the file exists in that directory then process
        projection_data = _calculate_projection_data(path)
        if projection_data:
            results.append([state] + projection_data)
    return pd.DataFrame(results, columns=CALCULATED_PROJECTION_HEADERS_STATES)


def get_county_projections_df(input_dir, intervention_type):
    """
    for each state in our data look at the results we generated via run.py
    to create the projections
    """
    fips_pd = FIPSPopulation.local().data  # to get the state, county & fips

    # save results in a list of lists, converted to df later
    results = []

    # get the state and fips so we can get the files
    missing = 0
    for index, fips_row in fips_pd.iterrows():
        state = fips_row["state"]
        fips = fips_row["fips"]

        file_name = f"{state}.{fips}.{intervention_type}.json"
        path = os.path.join(input_dir, "county", file_name)
        # if the file exists in that directory then process
        projection_data = _calculate_projection_data(path)
        if projection_data:
            results.append([state, fips] + projection_data)
        else:
            missing = missing + 1
    print(f"Models missing for {missing} counties")
    ndf = pd.DataFrame(results, columns=CALCULATED_PROJECTION_HEADERS_COUNTIES)
    return ndf
