import os
import logging
import urllib.request
import requests
import re
import io
import us
import zipfile
import json
from datetime import datetime
from functools import lru_cache
from enum import Enum

import pandas as pd
import numpy as np

from covidactnow.datapublic.common_fields import CommonFields
from libs.datasets import combined_datasets
from libs.datasets.timeseries import TimeseriesDataset
from libs.datasets.dataset_utils import AggregationLevel
from pyseir.utils import get_run_artifact_path, RunArtifact, ewma_smoothing

log = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pyseir_data")


class HospitalizationCategory(Enum):
    HOSPITALIZED = "hospitalized"
    ICU = "icu"

    def __str__(self):
        return str(self.value)


class HospitalizationDataType(Enum):
    CUMULATIVE_HOSPITALIZATIONS = "cumulative_hospitalizations"
    CURRENT_HOSPITALIZATIONS = "current_hospitalizations"


def load_zip_get_file(url, file, decoder="utf-8"):
    """
    Load a zipfile from a URL and extract a single file.  Note that this is
    not ideal and may fail for large files since the files must fit in memory.

    Parameters
    ----------
    url: str
        URL to read from.
    file: str
        Filename to pull out of the zipfile.
    decoder: str
        Usually None for raw bytes or 'utf-8', or 'latin1'

    Returns
    -------
    file_buffer: io.BytesIO or io.StringIO
        The file buffer for the requested file if decoder is None else return
        a decoded StringIO.
    """
    remotezip = urllib.request.urlopen(url)
    zipinmemory = io.BytesIO(remotezip.read())
    zf = zipfile.ZipFile(zipinmemory)
    byte_string = zf.read(file)
    if decoder:
        string = byte_string.decode(decoder)
        return io.StringIO(string)
    else:
        return io.BytesIO(byte_string)


def cache_mobility_data():
    """
    Pulled from https://github.com/descarteslabs/DL-COVID-19
    """
    log.info("Downloading mobility data.")
    url = "https://raw.githubusercontent.com/descarteslabs/DL-COVID-19/master/DL-us-mobility-daterow.csv"

    dtypes_mapping = {
        "country_code": str,
        "admin_level": int,
        "admin1": str,
        "admin2": str,
        "fips": str,
        "samples": int,
        "m50": float,
        "m50_index": float,
    }

    df = pd.read_csv(filepath_or_buffer=url, parse_dates=["date"], dtype=dtypes_mapping)
    df__m50 = df.query("admin_level == 2")[["fips", "date", "m50"]]
    df__m50_index = df.query("admin_level == 2")[["fips", "date", "m50_index"]]
    df__m50__final = df__m50.groupby("fips").agg(list).reset_index()
    df__m50_index__final = df__m50_index.groupby("fips").agg(list).reset_index()
    df__m50__final["m50"] = df__m50__final["m50"].apply(lambda x: np.array(x))
    df__m50_index__final["m50_index"] = df__m50_index__final["m50_index"].apply(
        lambda x: np.array(x)
    )

    df__m50__final.to_pickle(os.path.join(DATA_DIR, "mobility_data__m50.pkl"))
    df__m50_index__final.to_pickle(os.path.join(DATA_DIR, "mobility_data__m50_index.pkl"))


def cache_public_implementations_data():
    """
    Pulled from https://github.com/JieYingWu/COVID-19_US_County-level_Summaries
    """
    log.info("Downloading public implementations data")
    url = "https://raw.githubusercontent.com/JieYingWu/COVID-19_US_County-level_Summaries/master/raw_data/national/public_implementations_fips.csv"

    data = requests.get(url, verify=True).content.decode("utf-8")
    data = re.sub(r",(\d+)-(\w+)", r",\1-\2-2020", data)  # NOTE: This assumes the year 2020

    date_cols = [
        "stay at home",
        ">50 gatherings",
        ">500 gatherings",
        "public schools",
        "restaurant dine-in",
        "entertainment/gym",
        "Federal guidelines",
        "foreign travel ban",
    ]
    df = pd.read_csv(io.StringIO(data), parse_dates=date_cols, dtype="str").drop(
        ["Unnamed: 1", "Unnamed: 2"], axis=1
    )
    df.columns = [
        col.replace(">", "").replace(" ", "_").replace("/", "_").lower() for col in df.columns
    ]
    df.fips = df.fips.apply(lambda x: x.zfill(5))
    df.to_pickle(os.path.join(DATA_DIR, "public_implementations_data.pkl"))


@lru_cache(maxsize=32)
def load_county_metadata():
    """
    Return county level metadata such as age distributions, populations etc..

    Returns
    -------
    : pd.DataFrame

    """

    county_metadata = pd.read_json(
        os.path.join(DATA_DIR, "county_metadata.json"), dtype={"fips": "str"}
    )
    # Fix state names
    county_metadata.loc[:, "state"] = county_metadata["fips"].apply(
        lambda x: us.states.lookup(x[:2]).name
    )
    return county_metadata


@lru_cache(maxsize=32)
def load_county_metadata_by_state(state=None):
    """
    Generate a dataframe that contains county metadata aggregated at state
    level.

    Parameters
    ----------
    state: str or list(str)
        Name of state to load the metadata for.

    Returns
    -------
    state_metadata: pd.DataFrame
    """

    # aggregate into state level metadata
    state_metadata = load_county_metadata()

    if state is not None:
        state = [state] if isinstance(state, str) else list(state)
    else:
        state = state_metadata["state"].unique()

    state = [s.title() for s in state]

    state_metadata = state_metadata[state_metadata.state.isin(state)]

    density_measures = ["housing_density", "population_density"]
    for col in density_measures:
        state_metadata.loc[:, col] = state_metadata[col] * state_metadata["total_population"]

    age_dist = state_metadata.groupby("state")["age_distribution"].apply(
        lambda l: np.stack(np.array(l)).sum(axis=0)
    )
    density_info = state_metadata.groupby("state").agg(
        {
            "population_density": lambda x: sum(x),
            "housing_density": lambda x: sum(x),
            "total_population": lambda x: sum(x),
            "fips": list,
        }
    )
    age_bins = state_metadata[["state", "age_bin_edges"]].groupby("state").first()
    state_metadata = pd.concat([age_dist, density_info, age_bins], axis=1)

    for col in density_measures:
        state_metadata[col] /= state_metadata["total_population"]

    return state_metadata


@lru_cache(maxsize=32)
def load_ensemble_results(fips):
    """
    Retrieve ensemble results for a given state or county fips code.

    Parameters
    ----------
    fips: str
        State or county FIPS to load.

    Returns
    -------
    ensemble_results: dict
    """
    output_filename = get_run_artifact_path(fips, RunArtifact.ENSEMBLE_RESULT)
    if not os.path.exists(output_filename):
        return None

    with open(output_filename) as f:
        return json.load(f)


@lru_cache(maxsize=32)
def load_county_metadata_by_fips(fips):
    """
    Generate a dictionary for a county which includes county metadata.

    Parameters
    ----------
    fips: str

    Returns
    -------
    county_metadata: dict
        Dictionary of metadata for the county. The keys are:

        ['state', 'county', 'total_population', 'population_density',
        'housing_density', 'age_distribution', 'age_bin_edges']
    """
    county_metadata = load_county_metadata()
    county_metadata_merged = county_metadata.set_index("fips").loc[fips].to_dict()
    for key, value in county_metadata_merged.items():
        if np.isscalar(value) and not isinstance(value, str):
            county_metadata_merged[key] = float(value)
    return county_metadata_merged


@lru_cache(maxsize=32)
def get_all_fips_codes_for_a_state(state: str):
    """Returns a list of fips codes for a state

    Arguments:
        state {str} -- the full state name

    Returns:
        fips [list] -- a list of fips codes for a state
    """
    df = load_county_metadata()
    all_fips = df[df["state"].str.lower() == state.lower()].fips
    return all_fips


@lru_cache(maxsize=32)
def load_new_case_data_by_fips(
    fips, t0, include_testing_correction=False, testing_correction_smoothing_tau=5
):
    """
    Get data for new cases.

    Parameters
    ----------
    fips: str
        County fips to lookup.
    t0: datetime
        Datetime to offset by.
    include_testing_correction: bool
        If True, include a correction for new expanded or decreaseed test
        coverage.
    testing_correction_smoothing_tau: float
        expected_positives_from_test_increase is smoothed based on an
        exponentially weighted moving average of decay factor specified here.

    Returns
    -------
    times: array(float)
        List of float days since t0 for the case and death counts below
    observed_new_cases: array(int)
        Array of new cases observed each day.
    observed_new_deaths: array(int)
        Array of new deaths observed each day.
    """
    county_case_timeseries = combined_datasets.get_timeseries_for_fips(
        fips, columns=[CommonFields.CASES, CommonFields.DEATHS], min_range_with_some_value=True
    )
    county_case_data = county_case_timeseries.data

    times_new = (county_case_data["date"] - t0).dt.days.iloc[1:]
    observed_new_cases = (
        county_case_data["cases"].values[1:] - county_case_data["cases"].values[:-1]
    )

    if include_testing_correction:
        df_new_tests = load_new_test_data_by_fips(
            fips, t0, smoothing_tau=testing_correction_smoothing_tau
        )
        df_cases = pd.DataFrame({"times": times_new, "new_cases": observed_new_cases})
        df_cases = df_cases.merge(df_new_tests, how="left", on="times")
        df_cases["new_cases"] -= df_cases["expected_positives_from_test_increase"].fillna(0)
        observed_new_cases = df_cases["new_cases"].values

    observed_new_deaths = (
        county_case_data["deaths"].values[1:] - county_case_data["deaths"].values[:-1]
    )

    # Clip because there are sometimes negatives either due to data reporting or
    # corrections in case count. These are always tiny so we just make
    # downstream easier to work with by clipping.
    return times_new, observed_new_cases.clip(min=0), observed_new_deaths.clip(min=0)


@lru_cache(maxsize=32)
def load_new_case_data_by_state(
    state, t0, include_testing_correction=False, testing_correction_smoothing_tau=5
):
    """
    Get data for new cases at state level.

    Parameters
    ----------
    state: str
        State full name.
    t0: datetime
        Datetime to offset by.
    include_testing_correction: bool
        If True, include a correction for new expanded or decreaseed test
        coverage.
    testing_correction_smoothing_tau: float
        expected_positives_from_test_increase is smoothed based on an
        exponentially weighted moving average of decay factor specified here.

    Returns
    -------
    times: array(float)
        List of float days since t0 for the case and death counts below
    observed_new_cases: array(int)
        Array of new cases observed each day.
    observed_new_deaths: array(int)
        Array of new deaths observed each day.
    """
    state_abbrev = us.states.lookup(state).abbr
    state_timeseries = combined_datasets.get_timeseries_for_state(
        state_abbrev,
        columns=[CommonFields.CASES, CommonFields.DEATHS],
        min_range_with_some_value=True,
    )
    state_case_data = state_timeseries.data

    times_new = (state_case_data[CommonFields.DATE] - t0).dt.days.iloc[1:]
    observed_new_cases = (
        state_case_data[CommonFields.CASES].values[1:]
        - state_case_data[CommonFields.CASES].values[:-1]
    )

    if include_testing_correction:
        df_new_tests = load_new_test_data_by_fips(
            us.states.lookup(state).fips, t0, smoothing_tau=testing_correction_smoothing_tau
        )
        df_cases = pd.DataFrame({"times": times_new, "new_cases": observed_new_cases})
        df_cases = df_cases.merge(df_new_tests, how="left", on="times")
        df_cases["new_cases"] -= df_cases["expected_positives_from_test_increase"].fillna(0)
        observed_new_cases = df_cases["new_cases"].values

    observed_new_deaths = (
        state_case_data[CommonFields.DEATHS].values[1:]
        - state_case_data[CommonFields.DEATHS].values[:-1]
    )

    return (times_new, np.array(observed_new_cases).clip(min=0), observed_new_deaths.clip(min=0))


def get_hospitalization_data():
    """
    Since we're using this data for hospitalized data only, only returning
    values with hospitalization data.  I think as the use cases of this data source
    expand, we may not want to drop. For context, as of 4/8 607/1821 rows contained
    hospitalization data.
    Returns
    -------
    TimeseriesDataset
    """
    data = combined_datasets.build_us_timeseries_with_all_fields().data
    has_current_hospital = data[CommonFields.CURRENT_HOSPITALIZED].notnull()
    has_cumulative_hospital = data[CommonFields.CUMULATIVE_HOSPITALIZED].notnull()
    return TimeseriesDataset(data[has_current_hospital | has_cumulative_hospital])


@lru_cache(maxsize=32)
def load_hospitalization_data(
    fips: str,
    t0: datetime,
    category: HospitalizationCategory = HospitalizationCategory.HOSPITALIZED,
):
    """
    Obtain hospitalization data. We clip because there are sometimes negatives
    either due to data reporting or corrections in case count. These are always
    tiny so we just make downstream easier to work with by clipping.

    Parameters
    ----------
    fips: str
        County fips to lookup.
    t0: datetime
        Datetime to offset by.
    category: HospitalizationCategory

    Returns
    -------
    relative_days: array(float)
        List of float days since t0 for the hospitalization data.
    observed_hospitalizations: array(int)
        Array of new cases observed each day.
    type: HospitalizationDataType
        Specifies cumulative or current hospitalizations.
    """
    hospitalization_data = get_hospitalization_data().get_data(
        AggregationLevel.COUNTY, country="USA", fips=fips
    )

    if len(hospitalization_data) == 0:
        return None, None, None

    if (hospitalization_data[f"current_{category}"] > 0).any():
        hospitalization_data = hospitalization_data[
            hospitalization_data[f"current_{category}"].notnull()
        ]
        relative_days = (hospitalization_data["date"].dt.date - t0.date()).dt.days.values
        return (
            relative_days,
            hospitalization_data[f"current_{category}"].values.clip(min=0),
            HospitalizationDataType.CURRENT_HOSPITALIZATIONS,
        )
    elif (hospitalization_data[f"cumulative_{category}"] > 0).any():
        hospitalization_data = hospitalization_data[
            hospitalization_data[f"cumulative_{category}"].notnull()
        ]
        relative_days = (hospitalization_data["date"].dt.date - t0.date()).dt.days.values
        cumulative = hospitalization_data[f"cumulative_{category}"].values.clip(min=0)
        # Some minor glitches for a few states..
        for i, val in enumerate(cumulative[1:]):
            if cumulative[i] > cumulative[i + 1]:
                cumulative[i] = cumulative[i + 1]
        return relative_days, cumulative, HospitalizationDataType.CUMULATIVE_HOSPITALIZATIONS
    else:
        return None, None, None


@lru_cache(maxsize=32)
def load_hospitalization_data_by_state(
    state: str,
    t0: datetime,
    category: HospitalizationCategory = HospitalizationCategory.HOSPITALIZED,
):
    """
    Obtain hospitalization data. We clip because there are sometimes negatives
    either due to data reporting or corrections in case count. These are always
    tiny so we just make downstream easier to work with by clipping.

    Parameters
    ----------
    state: str
        State to lookup.
    t0: datetime
        Datetime to offset by.
    category: HospitalizationCategory
        'icu' for just ICU or 'hospitalized' for all ICU + Acute.

    Returns
    -------
    times: array(float) or NoneType
        List of float days since t0 for the hospitalization data.
    observed_hospitalizations: array(int) or NoneType
        Array of new cases observed each day.
    type: HospitalizationDataType
        Specifies cumulative or current hospitalizations.
    """
    abbr = us.states.lookup(state).abbr
    hospitalization_data = combined_datasets.build_us_timeseries_with_all_fields().get_data(
        AggregationLevel.STATE, country="USA", state=abbr
    )

    if len(hospitalization_data) == 0:
        return None, None, None

    if (hospitalization_data[f"current_{category}"] > 0).any():
        hospitalization_data = hospitalization_data[
            hospitalization_data[f"current_{category}"].notnull()
        ]
        times_new = (hospitalization_data["date"].dt.date - t0.date()).dt.days.values
        return (
            times_new,
            hospitalization_data[f"current_{category}"].values.clip(min=0),
            HospitalizationDataType.CURRENT_HOSPITALIZATIONS,
        )
    elif (hospitalization_data[f"cumulative_{category}"] > 0).any():
        hospitalization_data = hospitalization_data[
            hospitalization_data[f"cumulative_{category}"].notnull()
        ]
        times_new = (hospitalization_data["date"].dt.date - t0.date()).dt.days.values
        cumulative = hospitalization_data[f"cumulative_{category}"].values.clip(min=0)
        # Some minor glitches for a few states..
        for i, val in enumerate(cumulative[1:]):
            if cumulative[i] > cumulative[i + 1]:
                cumulative[i] = cumulative[i + 1]
        return (
            times_new,
            hospitalization_data[f"cumulative_{category}"].values.clip(min=0),
            HospitalizationDataType.CUMULATIVE_HOSPITALIZATIONS,
        )
    else:
        return None, None, None


def get_current_hospitalized(fips, t0, category: HospitalizationCategory):
    """
    Return the current estimate for the number of people in the given category for a given fips.
    Treats a length 2 fips as a state and a length 5 fips as a county

    Parameters
    ----------
    fips: str
        US fips to lookup.
    t0: datetime
        Datetime to offset by.
    category: HospitalizationCategory
        'icu' for just ICU or 'hospitalized' for all ICU + Acute.

    Returns
    -------
    time: float
        Days since t0 for the hospitalization data.
    current estimate: float
        The most recent provided value for the current occupied in the requested category.
    """
    df = combined_datasets.get_timeseries_for_fips(fips).data
    return _get_current_hospitalized(df, t0, category)


def _get_current_hospitalized(
    df: pd.DataFrame, t0: datetime, category: HospitalizationCategory,
):
    """
    Given a DataFrame that contains values icu or hospitalization data
    for a single county/state, this function returns the latest value.

    Parameters
    ----------
    df
        dataframe containing either current_ or cumulative_ values for a single county or state
    t0
        beginning of observation period
    category
        the type of current data to be returned

    Returns
    -------
    time: float
        Days since t0 for the hospitalization data.
    current estimate: float
        The most recent provided value for the current occupied in the requested category.
    """

    # TODO: No need to pass t0 down and back up. Can return a datetime that consumer converts.

    NUM_DAYS_LOOKBACK = 3
    # Agencies will start and stop reporting values. Also, depending on the time of day some columns
    # in a day row may propagate before others. Therefore, we don't want to just take the most
    # recent value which may be None, nor take just the most recent any value, which may be weeks
    # ago.

    # Datetimes are in naive but UTC. Look at possible values that are within this time window.
    date_minimum = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=NUM_DAYS_LOOKBACK)
    date_mask = df["date"] >= date_minimum.to_datetime64()
    recent_days_index = df.index[date_mask]

    # Look back from most recent and find the first (latest) non-null value. If the loop drops out,
    # that means there were no non-null values in the window of interest, and we return Nones.
    for idx in reversed(recent_days_index):  # Iterate from most recent backwards
        if pd.notnull(df[f"current_{category}"][idx]):
            current_latest = df[f"current_{category}"][idx]

            times_new = df["date"].dt.date - t0.date()
            times_new_latest = times_new[idx].days
            return times_new_latest, current_latest
    else:  # No values found in recent window, so return None
        return None, None


@lru_cache(maxsize=32)
def load_new_test_data_by_fips(fips, t0, smoothing_tau=5, correction_threshold=5):
    """
    Return a timeseries of new tests for a geography. Note that due to reporting
    discrepancies county to county, and state-to-state, these often do not go
    back as far as case data.

    Parameters
    ----------
    fips: str
        State or county fips code
    t0: datetime
        Reference datetime to use.

    Returns
    -------
    df: pd.DataFrame
        DataFrame containing columns:
        - 'date',
        - 'new_tests': Number of total tests performed that day
        - 'increase_in_new_tests': Increase in tests performed that day vs
          previous day
        - 'positivity_rate':
            Test positivity rate
        - 'expected_positives_from_test_increase':
            Number of positive detections expected just from increased test
            capacity.
        - times: days since t0 for this observation.
    smoothing_tau: int
        expected_positives_from_test_increase is smoothed based on an
        exponentially weighted moving average of decay factor specified here.
    correction_threshold: int
        Do not apply a correction if the incident cases per day is lower than
        this value. There can be instability if case counts are very low.
    """
    fips_timeseries = combined_datasets.get_timeseries_for_fips(fips)
    df = fips_timeseries.data.copy()

    # Aggregation level is None as fips is unique across aggregation levels.
    df = df.loc[
        (df[CommonFields.POSITIVE_TESTS].notnull())
        & (df[CommonFields.NEGATIVE_TESTS].notnull())
        & ((df[CommonFields.POSITIVE_TESTS] + df[CommonFields.NEGATIVE_TESTS]) > 0),
        :,
    ]

    df["positivity_rate"] = df[CommonFields.POSITIVE_TESTS] / (
        df[CommonFields.POSITIVE_TESTS] + df[CommonFields.NEGATIVE_TESTS]
    )
    df["new_positive"] = np.append([0], np.diff(df[CommonFields.POSITIVE_TESTS]))

    # The first derivative gets us new instead of cumulative tests while the second derivative gives us the change in new test rate.
    df["new_tests"] = np.append(
        [0], np.diff(df[CommonFields.POSITIVE_TESTS] + df[CommonFields.NEGATIVE_TESTS])
    )
    df["increase_in_new_tests"] = np.append([0], np.diff(df["new_tests"]))

    # dPositive / dTotal = 0.65 * positivity_rate was empirically determined by looking at
    # the increase in positives day-over-day relative to the increase in total tests across all 50 states.
    df["expected_positives_from_test_increase"] = (
        df["increase_in_new_tests"] * 0.65 * df["positivity_rate"]
    )
    df = df[
        [
            "date",
            "new_tests",
            "increase_in_new_tests",
            "positivity_rate",
            "expected_positives_from_test_increase",
            "new_positive",
        ]
    ]
    df = df.loc[df.increase_in_new_tests.notnull() & df.positivity_rate.notnull(), :]
    df["expected_positives_from_test_increase"] = ewma_smoothing(
        df["expected_positives_from_test_increase"], smoothing_tau
    )
    df.loc[df["new_positive"] < 5, "expected_positives_from_test_increase"] = 0

    df["times"] = [
        int((date - t0).days) for date in pd.to_datetime(df["date"].values).to_pydatetime()
    ]

    return df


def load_cdc_hospitalization_data():
    """
    Return age specific hospitalization rate.
    Source: https://www.cdc.gov/mmwr/volumes/69/wr/mm6912e2.htm#T1_down
    Table has columns: lower_age, upper_age, mean_age, lower_{outcome type},
    upper_{outcome type}, and mean_{outcome type}.
    Outcome types and their meanings:
    - hosp: percentage of all hospitalizations among cases
    - icu: percentage of icu admission among cases
    - hgen: percentage of general hospitalization (all hospitalizations - icu)
    - fatality: case fatality rate
    """

    return pd.read_csv(os.path.join(DATA_DIR, "cdc_hospitalization_data.csv"))


@lru_cache(maxsize=1)
def load_mobility_data_m50():
    """
    Return mobility data without normalization

    Returns
    -------
    : pd.DataFrame
    """
    return pd.read_pickle(os.path.join(DATA_DIR, "mobility_data__m50.pkl"))


@lru_cache(maxsize=1)
def load_mobility_data_m50_index():
    """
    Return mobility data with normalization: per
    https://github.com/descarteslabs/DL-COVID-19 normal m50 is defined during
    2020-02-17 to 2020-03-07.

    Returns
    -------
    : pd.DataFrame
    """
    return pd.read_pickle(os.path.join(DATA_DIR, "mobility_data__m50_index.pkl")).set_index("fips")


@lru_cache(maxsize=1)
def load_public_implementations_data():
    """
    Return public implementations data

    Returns
    -------
    : pd.DataFrame
    """
    return pd.read_pickle(os.path.join(DATA_DIR, "public_implementations_data.pkl")).set_index(
        "fips"
    )


def load_contact_matrix_data_by_fips(fips):
    """
    Load contact matrix for given fips.
    Source: polymod survey in UK
    (https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.0050074).
    Contact matrix at each county has been adjusted by county demographics.

    Parameters
    ----------
    fips: str
         State or county FIPS code.

    Returns
    -------
      : dict
        With fips as keys and values:
           - 'contact_matrix': list(list)
              number of contacts made by age group in rows with age groups in
              columns
           - 'age_bin_edges': list
              lower age limits to define age groups
           - 'age_distribution': list
             population size of each age group
    """

    fips = [fips] if isinstance(fips, str) else list(fips)
    state_abbr = us.states.lookup(fips[0][:2]).abbr
    path = os.path.join(DATA_DIR, "contact_matrix", "contact_matrix_fips_%s.json" % state_abbr)
    contact_matrix_data = json.loads(open(path).read())
    return {s: contact_matrix_data[s] for s in fips}


def load_whitelist():
    """
    Load the whitelist result.

    Returns
    -------
    whitelist: pd.DataFrame
        DataFrame containing a whitelist of product features for counties.
    """
    # Whitelist path isn't state specific, but the call requires ANY fips
    PLACEHOLDER_FIPS = "06"
    path = get_run_artifact_path(fips=PLACEHOLDER_FIPS, artifact=RunArtifact.WHITELIST_RESULT)
    return pd.read_json(path, dtype={"fips": str})


def cache_all_data():
    """
    Download all datasets locally.
    """
    cache_mobility_data()
    cache_public_implementations_data()


def get_compartment_value_on_date(fips, compartment, date, ensemble_results=None):
    """
    Return the value of compartment at a specified date.

    Parameters
    ----------
    fips: str
        State or County fips.
    compartment: str
        Name of the compartment to retrieve.
    date: datetime
        Date to retrieve values for.
    ensemble_results: NoneType or dict
        Pass in the pre-loaded simulation data to save time, else load it.
        Pass in the pre-loaded simulation data to save time, else load it.

    Returns
    -------
    value: float
        Value of compartment on a given date.
    """
    if ensemble_results is None:
        ensemble_results = load_ensemble_results(fips)
    # Circular import avoidance
    from pyseir.inference.fit_results import load_inference_result

    simulation_start_date = datetime.fromisoformat(load_inference_result(fips)["t0_date"])
    date_idx = int((date - simulation_start_date).days)
    return ensemble_results["suppression_policy__inferred"][compartment]["ci_50"][date_idx]


if __name__ == "__main__":
    cache_all_data()
