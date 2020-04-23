import pandas as pd
import numpy as np
import requests
import datetime
import os.path
import pprint
import shapefile
import simplejson
import statistics
import logging
import math, sys

from urllib.parse import urlparse
from collections import defaultdict
from functools import lru_cache

from libs.CovidDatasets import get_public_data_base_url
from libs.us_state_abbrev import US_STATE_ABBREV, us_fips
from libs.datasets import FIPSPopulation
from libs.datasets import JHUDataset
from libs.datasets import CovidTrackingDataSource
from libs.enums import Intervention
from libs.functions.calculate_projections import (
    get_state_projections_df,
    get_county_projections_df,
)
from libs.datasets.projections_schema import OUTPUT_COLUMN_REMAP_TO_RESULT_DATA
from libs.datasets.results_schema import (
    RESULT_DATA_COLUMNS_STATES,
    RESULT_DATA_COLUMNS_COUNTIES,
    CUMULATIVE_POSITIVE_TESTS,
    CUMULATIVE_NEGATIVE_TESTS,
)
from libs.constants import NULL_VALUE

_logger = logging.getLogger(__name__)


@lru_cache(None)
def _get_interventions_df():
    # TODO: read this from a dataset class
    interventions_url = "https://raw.githubusercontent.com/covid-projections/covid-projections/master/src/assets/data/interventions.json"
    interventions = requests.get(interventions_url).json()
    return pd.DataFrame(list(interventions.items()), columns=["state", "intervention"])


def _get_abbrev_df():
    # TODO: read this from a dataset class
    return pd.DataFrame(
        list(US_STATE_ABBREV.items()), columns=["state", "abbreviation"]
    )


def _get_testing_df():
    # TODO: read this from a dataset class
    abbrev_df = _get_abbrev_df()

    ctd_df = CovidTrackingDataSource.local().data
    ctd_df = ctd_df[CovidTrackingDataSource.TEST_FIELDS]
    ctd_df = ctd_df.merge(
            abbrev_df,
            left_on='state',
            right_on='abbreviation',
            how="inner",
            suffixes=['_dropcol', '']
        ).drop("state_dropcol", axis=1)

    return ctd_df


def get_testing_timeseries_by_state(state):
    test_df = _get_testing_df()
    # use a string for dates
    test_df['date'] = test_df.date.apply(lambda x: x.strftime("%m/%d/%y"))
    # handle missing data
    test_df['positive'] = test_df['positive'].astype(int)
    test_df['negative'] = test_df['negative'].astype(int)

    state_test_df = test_df[test_df[CovidTrackingDataSource.Fields.STATE] == state]
    # just select state
    return state_test_df[CovidTrackingDataSource.TESTS_ONLY_FIELDS]



county_replace_with_null = {"Unassigned": NULL_VALUE}


def _get_usa_by_county_df():
    # TODO: read this from a dataset class
    latest_path = JHUDataset.latest_path()
    _logger.info(f"Loading latest JHU data from {latest_path}")
    raw_df = pd.read_csv(latest_path, dtype={"FIPS": str})
    raw_df["FIPS"] = raw_df["FIPS"].astype(str).str.zfill(5)

    column_mapping = {
        "Province_State": "Province/State",
        "Country_Region": "Country/Region",
        "Last_Update": "Last Update",
        "Lat": "Latitude",
        "Long_": "Longitude",
        "Combined_Key": "Combined Key",
        "Admin2": "County",
        "FIPS": "State/County FIPS Code",
    }
    remapped_df = raw_df.rename(columns=column_mapping)

    # USA only
    us_df = remapped_df[(remapped_df["Country/Region"] == "US")]
    jhu_column_names = [
        "Province/State",
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
    final_df = pd.DataFrame(us_df, columns=jhu_column_names)
    final_df["Last Update"] = pd.to_datetime(final_df["Last Update"])
    final_df["Last Update"] = final_df["Last Update"].dt.strftime("%-m/%-d/%Y %H:%M")

    final_df["County"] = final_df["County"].replace(county_replace_with_null)
    final_df["Combined Key"] = final_df["Combined Key"].str.replace("Unassigned, ", "")
    final_df = final_df.fillna(NULL_VALUE)
    final_df = final_df.drop_duplicates(
        "State/County FIPS Code"
    )  # note this is a hack, 49053 is dupped in JHU data :(
    final_df.index.name = "OBJECTID"
    # assert unique key test
    assert final_df["Combined Key"].value_counts().max() == 1
    assert final_df["State/County FIPS Code"].value_counts().max() == 1

    return final_df


def get_usa_by_county_with_projection_df(input_dir, intervention_type):
    us_only = _get_usa_by_county_df()
    fips_df = FIPSPopulation.local().data  # used to get interventions
    interventions_df = _get_interventions_df()
    projections_df = get_county_projections_df(
        input_dir, intervention_type, interventions_df
    )

    counties_decorated = (
        us_only.merge(
            projections_df,
            left_on="State/County FIPS Code",
            right_on="FIPS",
            how="inner",
        ).merge(
            fips_df[["state", "fips"]],
            left_on="FIPS", right_on="fips", how="inner"
        ).merge(
            interventions_df, left_on="state", right_on="state", how="inner"
        ).drop(['State', 'state'], axis=1)
    )

    counties_remapped = counties_decorated.rename(
        columns=OUTPUT_COLUMN_REMAP_TO_RESULT_DATA
    )
    counties = pd.DataFrame(counties_remapped)[RESULT_DATA_COLUMNS_COUNTIES]
    counties = counties.fillna(NULL_VALUE)
    counties.index.name = "OBJECTID"
    # assert unique key test

    if counties["Combined Key"].value_counts().max() != 1:
        raise Exception(
            f"counties['Combined Key'].value_counts().max() = {counties['Combined Key'].value_counts().max()}, at input_dir {input_dir}."
        )
    return counties


def get_usa_by_states_df(input_dir, intervention_type):

    us_only = _get_usa_by_county_df()
    abbrev_df = _get_abbrev_df()
    interventions_df = _get_interventions_df()
    projections_df = get_state_projections_df(
        input_dir, intervention_type, interventions_df
    )
    test_df = _get_testing_df()
    test_max_df = test_df.groupby("abbreviation")[
        CovidTrackingDataSource.Fields.POSITIVE_TESTS,
        CovidTrackingDataSource.Fields.NEGATIVE_TESTS
    ].max().reset_index()

    states_group = us_only.groupby(["Province/State"])
    states_agg = states_group.aggregate(
        {
            "Last Update": "max",
            "Confirmed": "sum",
            "Recovered": "sum",
            "Deaths": "sum",
            "Active": "sum",
            "Country/Region": "first",
            "Latitude": "first",
            "Longitude": "first"
            # People tested is currently null
            #'People Tested': 'sum'
        }
    )

    # basically the states_agg has full state names, the interventions have abbreviation so we need these to be merged
    # inner merge to filter to only the 50 states+DC.  (left join to avoid missing data)
    states_abbrev = (
        states_agg.merge(
            abbrev_df,  # adds 'state', 'abbreviation'
            left_index=True,
            right_on="state",
            how="left",
        ).merge(
            test_max_df,  # adds state, positive, negative
            left_on="abbreviation",
            right_on="abbreviation",
            how="left",
        ).merge(
            interventions_df,  # add intervention columns
            left_on="abbreviation",
            right_on="state",
            how="inner",
            suffixes=["", "_dropcol"],
        ).drop(
            "state_dropcol", axis=1
        ).merge(
            projections_df,  # add projection columns
            left_on="abbreviation",
            right_on="State",
            how="left",
        ).drop(["abbreviation", "State"], axis=1)
    )

    STATE_COLS_REMAP = {
        CovidTrackingDataSource.Fields.POSITIVE_TESTS: CUMULATIVE_POSITIVE_TESTS,
        CovidTrackingDataSource.Fields.NEGATIVE_TESTS: CUMULATIVE_NEGATIVE_TESTS,
        **OUTPUT_COLUMN_REMAP_TO_RESULT_DATA
    }

    states_remapped = states_abbrev.rename(columns=STATE_COLS_REMAP)

    states_final = pd.DataFrame(states_remapped, columns=RESULT_DATA_COLUMNS_STATES)
    states_final[CUMULATIVE_POSITIVE_TESTS] = states_final[CUMULATIVE_POSITIVE_TESTS].fillna(0)
    states_final[CUMULATIVE_NEGATIVE_TESTS] = states_final[CUMULATIVE_NEGATIVE_TESTS].fillna(0)

    states_final = states_final.fillna(NULL_VALUE)
    states_final["Combined Key"] = states_final["Province/State"]
    states_final["State/County FIPS Code"] = states_final["Province/State"].map(us_fips)

    states_final.index.name = "OBJECTID"

    # assert unique key test
    assert states_final["Combined Key"].value_counts().max() == 1

    return states_final


# us_only = _get_usa_by_county_df()
# us_only.to_csv("results/counties.csv")

# states_final = get_usa_by_states_df()
# states_final.to_csv('results/states.csv')
