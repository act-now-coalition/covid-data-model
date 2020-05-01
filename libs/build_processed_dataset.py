import pandas as pd
import requests
import logging

from functools import lru_cache

from libs.us_state_abbrev import US_STATE_ABBREV
from libs.us_state_abbrev import abbrev_us_state
from libs.us_state_abbrev import us_fips


from libs.datasets import FIPSPopulation
from libs.datasets import JHUDataset
from libs.datasets import CovidTrackingDataSource
from libs.datasets import CDSDataset
from libs.datasets.common_fields import CommonFields
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
    columns = [CommonFields.STATE, "intervention"]
    return pd.DataFrame(list(interventions.items()), columns=columns)


@lru_cache(None)
def _get_testing_df():
    # TODO: read this from a dataset class
    ctd_df = CovidTrackingDataSource.local().data
    # use a string for dates
    ctd_df["date"] = ctd_df.date.apply(lambda x: x.strftime("%m/%d/%y"))
    # handle missing data
    ctd_df[CovidTrackingDataSource.Fields.POSITIVE_TESTS] = ctd_df[
        CovidTrackingDataSource.Fields.POSITIVE_TESTS
    ].apply(lambda x: x if pd.isna(x) else int(x))
    ctd_df[CovidTrackingDataSource.Fields.NEGATIVE_TESTS] = ctd_df[
        CovidTrackingDataSource.Fields.NEGATIVE_TESTS
    ].apply(lambda x: x if pd.isna(x) else int(x))
    ctd_df = ctd_df[CovidTrackingDataSource.TEST_FIELDS]
    return ctd_df


@lru_cache(None)
def get_cds():
    cds_df = CDSDataset.local().data
    cds_df["date"] = cds_df.date.apply(lambda x: x.strftime("%m/%d/%y"))
    cds_df = cds_df[CDSDataset.TEST_FIELDS]
    return cds_df


def get_testing_timeseries_by_state(state):
    testing_df = _get_testing_df()
    # just select state
    state_testing_df = testing_df[
        testing_df[CovidTrackingDataSource.Fields.STATE] == state
    ]
    return state_testing_df[CovidTrackingDataSource.TESTS_ONLY_FIELDS]


def get_testing_timeseries_by_fips(fips):
    testing_df = get_cds()
    # select by fips
    fips_testing_df = testing_df[
        testing_df[CDSDataset.Fields.FIPS] == fips
    ]
    return fips_testing_df


county_replace_with_null = {"Unassigned": NULL_VALUE}


def _get_usa_by_county_df():
    # TODO: read this from a dataset class
    latest_path = JHUDataset.latest_path()
    _logger.info(f"Loading latest JHU data from {latest_path}")
    raw_df = pd.read_csv(latest_path, dtype={"FIPS": str})
    raw_df["FIPS"] = raw_df["FIPS"].astype(str).str.zfill(5)

    column_mapping = {
        "Province_State": CommonFields.STATE_FULL_NAME,
        "Country_Region": "Country/Region",
        "Last_Update": "Last Update",
        "Lat": "Latitude",
        "Long_": "Longitude",
        "Combined_Key": "Combined Key",
        "Admin2": "County",
        "FIPS": CommonFields.FIPS,
    }
    remapped_df = raw_df.rename(columns=column_mapping)
    # USA only
    us_df = remapped_df[(remapped_df["Country/Region"] == "US")]
    jhu_column_names = [
        CommonFields.STATE_FULL_NAME,
        "Country/Region",
        "Last Update",
        "Latitude",
        "Longitude",
        "Confirmed",
        "Recovered",
        "Deaths",
        "Active",
        "County",
        CommonFields.FIPS,
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

    final_df[CommonFields.STATE] = final_df[CommonFields.STATE_FULL_NAME].map(
        US_STATE_ABBREV
    )

    final_df = final_df.fillna(NULL_VALUE)
    # note this is a hack, 49053 is dupped in JHU data :(
    final_df = final_df.drop_duplicates(CommonFields.FIPS)
    final_df.index.name = "OBJECTID"
    # assert unique key test
    assert final_df["Combined Key"].value_counts().max() == 1
    assert final_df[CommonFields.FIPS].value_counts().max() == 1

    return final_df


def get_usa_by_county_with_projection_df(input_dir, intervention_type):
    us_only = _get_usa_by_county_df()
    fips_df = FIPSPopulation.local().data  # used to get interventions
    interventions_df = _get_interventions_df()
    projections_df = get_county_projections_df(
        input_dir, intervention_type, interventions_df
    )
    counties_decorated = (
        us_only.merge(projections_df, on=CommonFields.FIPS, how="inner")
        .merge(
            fips_df[[CommonFields.STATE, CommonFields.FIPS]],
            on=CommonFields.FIPS,
            how="inner",
        )
        .merge(interventions_df, on=CommonFields.STATE, how="inner")
    )
    counties_remapped = counties_decorated.rename(
        columns=OUTPUT_COLUMN_REMAP_TO_RESULT_DATA
    )
    counties = pd.DataFrame(counties_remapped)[RESULT_DATA_COLUMNS_COUNTIES]
    counties = counties.fillna(NULL_VALUE)
    counties.index.name = "OBJECTID"

    if counties["Combined Key"].value_counts().max() != 1:
        combined_key_max = counties["Combined Key"].value_counts().max()
        raise Exception(
            "counties['Combined Key'].value_counts().max() = "
            f"{combined_key_max}, at input_dir {input_dir}."
        )
    return counties


def get_usa_by_states_df(input_dir, intervention_type):
    us_only = _get_usa_by_county_df()
    interventions_df = _get_interventions_df()
    projections_df = get_state_projections_df(
        input_dir, intervention_type, interventions_df
    )
    testing_df = _get_testing_df()
    test_max_df = (
        testing_df.groupby(CommonFields.STATE)[
            CovidTrackingDataSource.Fields.POSITIVE_TESTS,
            CovidTrackingDataSource.Fields.NEGATIVE_TESTS,
        ]
        .max()
        .reset_index()
    )
    states_group = us_only.groupby([CommonFields.STATE])
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

    states_abbrev = (
        states_agg.merge(test_max_df, on=CommonFields.STATE, how="left")
        .merge(
            interventions_df,
            on=CommonFields.STATE,
            how="inner",
            suffixes=["", "_dropcol"],
        )
        .merge(projections_df, on=CommonFields.STATE, how="left")
    )
    STATE_COLS_REMAP = {
        CovidTrackingDataSource.Fields.POSITIVE_TESTS: CUMULATIVE_POSITIVE_TESTS,
        CovidTrackingDataSource.Fields.NEGATIVE_TESTS: CUMULATIVE_NEGATIVE_TESTS,
        **OUTPUT_COLUMN_REMAP_TO_RESULT_DATA,
    }

    states_remapped = states_abbrev.rename(columns=STATE_COLS_REMAP)
    states_remapped[CommonFields.STATE_FULL_NAME] = states_remapped[
        CommonFields.STATE
    ].map(abbrev_us_state)
    states_final = pd.DataFrame(states_remapped, columns=RESULT_DATA_COLUMNS_STATES)

    # Keep nulls as nulls
    states_final = states_final.fillna(NULL_VALUE)
    states_final["Combined Key"] = states_final[CommonFields.STATE_FULL_NAME]
    states_final[CommonFields.FIPS] = states_final[CommonFields.STATE_FULL_NAME].map(
        us_fips
    )

    states_final.index.name = "OBJECTID"

    assert states_final["Combined Key"].value_counts().max() == 1
    return states_final
