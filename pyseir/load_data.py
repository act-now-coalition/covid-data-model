import os
import logging
import pandas as pd
import numpy as np
import urllib.request
import requests
import re
import io
import us
import zipfile
import json
from datetime import datetime
from libs.datasets import NYTimesDataset
from libs.datasets import combined_datasets
from libs.datasets.timeseries import TimeseriesDataset
from libs.datasets.dataset_utils import AggregationLevel
from libs.datasets import CovidTrackingDataSource
from pyseir.utils import get_run_artifact_path, RunArtifact
from functools import lru_cache
from enum import Enum


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pyseir_data")


class HospitalizationDataType(Enum):
    CUMULATIVE_HOSPITALIZATIONS = "cumulative_hospitalizations"
    CURRENT_HOSPITALIZATIONS = "current_hospitalizations"


def hampel_filter__low_outliers_only(input_series, window_size=5, n_sigmas=2):
    """
    Filter out points with median absolute deviation greater than n_sigma from a
    nearest set of window-size neighbors. This is a very conservative filter to
    clean out some case / death data like Arkansas.  We apply this only to drops
    in counts that should be positive (e.g. Arkansas).

    Parameters
    ----------
    input_series: array
    window_size: int
    n_sigmas: float

    Returns
    -------

    """
    n = len(input_series)
    new_series = input_series.copy()
    k = 1.4826  # scale factor for Gaussian distribution

    indices = []

    # possibly use np.nanmedian
    for i in range(window_size, n - window_size):
        x0 = np.median(input_series[(i - window_size) : (i + window_size)])
        S0 = k * np.median(np.abs(input_series[(i - window_size) : (i + window_size)] - x0))
        if -(input_series[i] - x0) > n_sigmas * S0:
            new_series[i] = x0
            indices.append(i)

    return new_series, indices


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


def cache_county_case_data():
    """
    Cache county covid case data from NYT in #PYSEIR_HOME/data.
    """
    logging.info("Downloading covid case data")
    # NYT dataset
    county_case_data = load_county_case_data()
    county_case_data.to_pickle(os.path.join(DATA_DIR, "covid_case_timeseries.pkl"))


def cache_mobility_data():
    """
    Pulled from https://github.com/descarteslabs/DL-COVID-19
    """
    logging.info("Downloading mobility data.")
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
    logging.info("Downloading public implementations data")
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
def load_county_case_data():
    """
    Return county level case data.

    Returns
    -------
    : pd.DataFrame
    """
    county_case_data = (
        NYTimesDataset.local().timeseries().get_data(AggregationLevel.COUNTY, country="USA")
    )
    return county_case_data


@lru_cache(maxsize=1)
def load_state_case_data():
    """
    Return county level case data.

    Returns
    -------
    : pd.DataFrame
    """

    state_case_data = (
        NYTimesDataset.local().timeseries().get_data(AggregationLevel.STATE, country="USA")
    )
    return state_case_data


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
    if os.path.exists(output_filename):
        with open(output_filename) as f:
            fit_results = json.load(f)
        return fit_results
    return None


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
def load_new_case_data_by_fips(fips, t0):
    """
    Get data for new cases.

    Parameters
    ----------
    fips: str
        County fips to lookup.
    t0: datetime
        Datetime to offset by.

    Returns
    -------
    times: array(float)
        List of float days since t0 for the case and death counts below
    observed_new_cases: array(int)
        Array of new cases observed each day.
    observed_new_deaths: array(int)
        Array of new deaths observed each day.
    """
    _county_case_data = load_county_case_data()
    county_case_data = _county_case_data[_county_case_data["fips"] == fips]
    times_new = (county_case_data["date"] - t0).dt.days.iloc[1:]
    observed_new_cases = (
        county_case_data["cases"].values[1:] - county_case_data["cases"].values[:-1]
    )
    observed_new_deaths = (
        county_case_data["deaths"].values[1:] - county_case_data["deaths"].values[:-1]
    )

    # Clip because there are sometimes negatives either due to data reporting or
    # corrections in case count. These are always tiny so we just make
    # downstream easier to work with by clipping.
    return times_new, observed_new_cases.clip(min=0), observed_new_deaths.clip(min=0)


def get_hospitalization_data():
    data = combined_datasets.build_us_timeseries_with_all_fields().data
    # Since we're using this data for hospitalized data only, only returning
    # values with hospitalization data.  I think as the use cases of this data source
    # expand, we may not want to drop. For context, as of 4/8 607/1821 rows contained
    # hospitalization data.
    has_current_hospital = data[TimeseriesDataset.Fields.CURRENT_HOSPITALIZED].notnull()
    has_cumulative_hospital = data[TimeseriesDataset.Fields.CUMULATIVE_HOSPITALIZED].notnull()
    return TimeseriesDataset(data[has_current_hospital | has_cumulative_hospital])


@lru_cache(maxsize=32)
def load_hospitalization_data(fips, t0, category="hospitalized"):
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
    category: str
        'icu' or 'hospitalized'

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
def load_hospitalization_data_by_state(state, t0, category="hospitalized"):
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
    category: str
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

    categories = ["icu", "hospitalized"]
    if category not in categories:
        raise ValueError(f"Hospitalization category {category} is not in {categories}")

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


def get_current_hospitalized(state, t0, category):
    """
    Return the current estimate for the number of people in the given category for a given US state.

    Parameters
    ----------
    state: str
        US state to lookup.
    t0: datetime
        Datetime to offset by.
    category: str
        'icu' for just ICU or 'hospitalized' for all ICU + Acute.

    Returns
    -------
    time: float
        Days since t0 for the hospitalization data.
    current estimate: float
        The most recent estimate for the current occupied in the requested category.
    """
    MIN_DATAPOINTS_TO_CONVERT = 3  # We wait until the third datapoint to have 2 deltas to forecast

    abbr = us.states.lookup(state).abbr
    df = combined_datasets.build_us_timeseries_with_all_fields().get_data(
        AggregationLevel.STATE, country="USA", state=abbr
    )

    categories = ["icu", "hospitalized"]
    if category not in categories:
        raise ValueError(f"Hospitalization category {category} is not in {categories}")

    if len(df) == 0:
        return None, None

    # If data available in current_{} column, then return latest not-null value
    if (df[f"current_{category}"] > 0).any():
        df = df[df[f"current_{category}"].notnull()]
        df_latest = df[f"current_{category}"].values.clip(min=0)[-1]
        times_new = (df["date"].dt.date - t0.date()).dt.days.values
        times_new_latest = times_new[-1]
        return times_new_latest, df_latest  # Return current since available

    # If data is available in cumulative, try to convert to current (not just daily)
    elif (df[f"cumulative_{category}"] > 0).any():
        # Remove Null & Enforce Monotonically Increasing Cumulatives
        df = df[df[f"cumulative_{category}"].notnull()]
        cumulative = df[f"cumulative_{category}"].values.clip(min=0)
        for i, val in enumerate(cumulative[1:]):
            if cumulative[i] > cumulative[i + 1]:
                cumulative[i] = cumulative[i + 1]

        # Estimate Current from Derived Dailies
        if len(cumulative) >= MIN_DATAPOINTS_TO_CONVERT:
            current_latest = estimate_current_from_cumulative(cumulative, category)
            times_new = (df["date"].dt.date - t0.date()).dt.days.values
            times_new_latest = times_new[-1]
            return times_new_latest, current_latest  # Return current estimate from cumulative
        else:
            return None, None  # No current, not enough cumulative
    else:
        return None, None  # No current nor cumulative


def estimate_current_from_cumulative(cumulative, category):
    """
    We assume that an agency starts reporting cumulative admissions at a time not related to a
    particularly abnormal patient admission. So we use the data we have collected so far, and
    extrapolate backwards to estimate the ICU population at the start of reporting (which is
    non-zero). This should significantly speed up settling time (and the only
    movements from then on will be from changing inputs).

    The simple model provided (x_{i+1} = x_{i} + new - x_{i}/avg_length_of_stay
    has a steady state solution with constant input of new * avg_length_of_stay.

    We initialize the model by taking the first X data points, calculating the average, and then
    calculating steady state if historical data had matched current data. X is the average
    length of stay. We then use that as the starting point to step forward in the model. So once we
    have more than X datapoints, the forecast will shift based on underlying changes in the data.
    Until we collect significant data, the estimates are sensitive to the inital values reported.
    E.g. If day 1 shows 10 new ICU patients, we assume that has been happening for the last week
    too and estimate accordingly. We can choose to wait for multiple points before extrapolating
    to protect against surfacing noisy initial data to the user.

    Parameters
    ----------
    cumulative: array
        Array like sequence of daily cumulative values (expects, but doesn't enforce monotonic)
    category: str
        Either 'hospitalization' or 'icu

    Returns
    -------
    current_estimate: float
        Latest estimate of currently occupied beds.
    """
    average_length_of_stay = get_average_dwell_time(category)

    # Calculate new admissions as the differences of daily cumulatives
    daily_admits = np.diff(cumulative)

    # When reporting starts, we assume there will already be a non-zero number of patients
    # in the hospital/ICU. We need to estimate that starting value to initialize the model.
    # If we don't, it takes ~2 times the average length of stay for the value to reach the
    # expected. So our reported numbers for the first ~14 days would be too low (which we saw in
    # Utah ICU data).

    # We average over the inputs for up to the same number of days as the average dwell
    max_window = int(np.floor(average_length_of_stay))
    initial_daily_estimate = daily_admits[:max_window].mean()

    # And use that steady state solution as if that data had been entered in the past to initialize.
    t0_patients = initial_daily_estimate * average_length_of_stay

    # Step through the model and generate a output
    current_pts = []
    for step, new_day in enumerate(daily_admits):
        if step == 0:
            yesterday = t0_patients
        else:
            yesterday = current_pts[-1]

        today = yesterday + new_day - yesterday / average_length_of_stay
        current_pts.append(today)
    current_pts_latest = current_pts[-1]
    return current_pts_latest


def get_average_dwell_time(category):
    """
    :parameter
    category: str
        Whether we are asking for 'hospital' or 'icu'

    :return:
    average_length_of_stay: float
        the average length of stay for a given category
    """
    # Must be here to avoid circular import. This is required to convert
    # cumulative hosps to current hosps. We also just use a dummy fips and t_list.
    from pyseir.parameters.parameter_ensemble_generator import ParameterEnsembleGenerator

    # Surprisingly long load time if initial call (14sec) but then fast (22ms)
    params = ParameterEnsembleGenerator(
        fips="06",
        t_list=[],
        N_samples=250  # We want close to the ensemble mean.
        # Eventually replace with constants derived from the mean characteristic.
        # Then we can revert back to 1.
    ).get_average_seir_parameters()
    # TODO: This value is temporarily in this limited scope.
    # Will be added into the params once I decide on some refactoring.
    params["hospitalization_length_of_stay_icu_avg"] = 8.6
    if category == "hospitalized":
        average_length_of_stay = (
            params["hospitalization_rate_general"]
            * params["hospitalization_length_of_stay_general"]
            + params["hospitalization_rate_icu"]
            * (1 - params["fraction_icu_requiring_ventilator"])
            * params["hospitalization_length_of_stay_icu"]
            + params["hospitalization_rate_icu"]
            * params["fraction_icu_requiring_ventilator"]
            * params["hospitalization_length_of_stay_icu_and_ventilator"]
        ) / (params["hospitalization_rate_general"] + params["hospitalization_rate_icu"])
    else:
        # This value is a weighted average of icu w & w/o ventilator.
        # It is deterministic. Warning: This param was added in this very local scope.
        average_length_of_stay = params["hospitalization_length_of_stay_icu_avg"]
    return average_length_of_stay


@lru_cache(maxsize=32)
def load_new_case_data_by_state(state, t0):
    """
    Get data for new cases at state level.

    Parameters
    ----------
    state: str
        State full name.
    t0: datetime
        Datetime to offset by.

    Returns
    -------
    times: array(float)
        List of float days since t0 for the case and death counts below
    observed_new_cases: array(int)
        Array of new cases observed each day.
    observed_new_deaths: array(int)
        Array of new deaths observed each day.
    """
    _state_case_data = load_state_case_data()
    state_case_data = _state_case_data[_state_case_data["state"] == us.states.lookup(state).abbr]
    times_new = (state_case_data["date"] - t0).dt.days.iloc[1:]
    observed_new_cases = state_case_data["cases"].values[1:] - state_case_data["cases"].values[:-1]
    observed_new_deaths = (
        state_case_data["deaths"].values[1:] - state_case_data["deaths"].values[:-1]
    )

    _, filter_idx = hampel_filter__low_outliers_only(observed_new_cases, window_size=5, n_sigmas=2)
    keep_idx = np.array([i for i in range(len(times_new)) if i not in list(filter_idx)])
    times_new = [int(list(times_new)[idx]) for idx in keep_idx]
    return (
        times_new,
        np.array(observed_new_cases[keep_idx]).clip(min=0),
        observed_new_deaths.clip(min=0)[keep_idx],
    )


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
    path = get_run_artifact_path(
        fips="06", artifact=RunArtifact.WHITELIST_RESULT  # dummy since not used for whitelist.
    )
    return pd.read_json(path, dtype={"fips": str})


def cache_all_data():
    """
    Download all datasets locally.
    """
    cache_county_case_data()
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
