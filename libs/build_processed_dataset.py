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

from libs.CovidDatasets import get_public_data_base_url
from libs.us_state_abbrev import us_state_abbrev, us_fips, abbrev_us_fips, abbrev_us_state
from libs.datasets import FIPSPopulation
from libs.datasets.dataset_utils import AggregationLevel
from libs.datasets import JHUDataset
from libs.datasets.timeseries import TimeseriesDataset
from libs.enums import Intervention
from libs.functions.calculate_projections import (
    get_state_projections_df,
    get_county_projections_df,
)
from libs.datasets.projections_schema import OUTPUT_COLUMN_REMAP_TO_RESULT_DATA
from libs.datasets.results_schema import (
    RESULT_DATA_COLUMNS_STATES,
    RESULT_DATA_COLUMNS_COUNTIES,
)
from libs.constants import NULL_VALUE

# @TODO: Attempt today. If that fails, attempt yesterday.
latest = datetime.date.today() - datetime.timedelta(days=1)


def _get_interventions_df():
    # TODO: read this from a dataset class
    interventions_url = "https://raw.githubusercontent.com/covid-projections/covid-projections/master/src/assets/data/interventions.json"
    interventions = requests.get(interventions_url).json()
    return pd.DataFrame(list(interventions.items()), columns=["state", "intervention"])


def _get_abbrev_df():
    # TODO: read this from a dataset class
    return pd.DataFrame(
        list(us_state_abbrev.items()), columns=["state", "abbreviation"]
    )


county_replace_with_null = {"Unassigned": NULL_VALUE}


def _get_usa_by_county_df():
    data = JHUDataset.local().timeseries()
    data = data.get_subset(None, country="USA")
    return data.latest_values(AggregationLevel.COUNTY)


def _get_usa_by_states_df():
    data = JHUDataset.local().timeseries()
    data = data.get_subset(None, country="USA")
    return data.latest_values(AggregationLevel.STATE)


def get_usa_by_county_with_projection_df(input_dir, intervention_type):
    print(input_dir, intervention_type)
    us_only = _get_usa_by_county_df()
    fips_df = FIPSPopulation.local().data
    # used to get interventions
    interventions_df = _get_interventions_df()
    projections_df = get_county_projections_df(
        input_dir, intervention_type, interventions_df
    )
    print(interventions_df.head())
    print(projections_df.head())
    print(us_only.head())
    print("HIIIII")

    counties_decorated = (
        us_only.merge(
            projections_df,
            left_on=TimeseriesDataset.Fields.FIPS,
            right_on="FIPS",
            how="inner",
        )
        .merge(interventions_df, left_on="state", right_on="state", how="inner")
    )
    counties_remapped = counties_decorated.rename(
        columns=OUTPUT_COLUMN_REMAP_TO_RESULT_DATA
    )
    counties = pd.DataFrame(counties_remapped, columns=RESULT_DATA_COLUMNS_COUNTIES)
    counties = counties.fillna(NULL_VALUE)
    counties.index.name = "OBJECTID"
    counties["Province/State"] = counties["Province/State"].map(abbrev_us_state)

    # assert unique key test
    assert counties["State/County FIPS Code"].value_counts().max() == 1

    return counties


def get_usa_by_states_df(input_dir, intervention_type):
    print(input_dir, intervention_type)
    us_only = _get_usa_by_states_df()
    interventions_df = _get_interventions_df()
    projections_df = get_state_projections_df(
        input_dir, intervention_type, interventions_df
    )
    # basically the states_agg has full state names, the interventions have
    # abbreviation so we need these to be merged
    states_abbrev = (
        us_only
        .merge(
            # inner merge to filter to only the 50 states
            interventions_df,
            left_on=TimeseriesDataset.Fields.STATE,
            right_on="state",
            how="inner",
        )
        .merge(projections_df, left_on="state", right_on="State", how="left")
    )
    states_remapped = states_abbrev.rename(columns=OUTPUT_COLUMN_REMAP_TO_RESULT_DATA)

    states_final = pd.DataFrame(states_remapped, columns=RESULT_DATA_COLUMNS_STATES)
    states_final = states_final.fillna(NULL_VALUE)
    states_final["Combined Key"] = states_final["Province/State"]
    # Don't have fips codes for states, need to map using state -> fips
    states_final["State/County FIPS Code"] = states_final["Province/State"].map(abbrev_us_fips)

    # States are output in abbreviated form, original output exects long form names.
    states_final["Province/State"] = states_final["Province/State"].map(abbrev_us_state)

    states_final.index.name = "OBJECTID"
    # assert unique key test
    assert states_final["Combined Key"].value_counts().max() == 1
    return states_final


# us_only = _get_usa_by_county_df()
# us_only.to_csv("results/counties.csv")

# states_final = get_usa_by_states_df()
# states_final.to_csv('results/states.csv')
