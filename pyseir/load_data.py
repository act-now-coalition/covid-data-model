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
import pyseir.utils

# from pyseir.utils import get_run_artifact_path, RunArtifact, ewma_smoothing

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
        fips_timeseries = combined_datasets.get_timeseries_for_fips(fips)
        df_new_tests = load_new_test_data_by_fips(
            fips_timeseries, t0, smoothing_tau=testing_correction_smoothing_tau
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
    data = combined_datasets.load_us_timeseries_dataset().data
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
    hospitalization_data = get_hospitalization_data().get_data(fips=fips)

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
        for i in range(cumulative[1:]):
            if cumulative[i] > cumulative[i + 1]:
                cumulative[i] = cumulative[i + 1]
        return relative_days, cumulative, HospitalizationDataType.CUMULATIVE_HOSPITALIZATIONS
    else:
        return None, None, None


@lru_cache(maxsize=32)
def load_new_test_data_by_fips(
    timeseries_dataset: TimeseriesDataset, t0, smoothing_tau=5, correction_threshold=5
):
    """
    Return a timeseries of new tests for a geography. Note that due to reporting
    discrepancies county to county, and state-to-state, these often do not go
    back as far as case data.
    Parameters
    ----------
    timeseries_dataset
        Data for the region
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
    df = timeseries_dataset.data.copy()

    # Aggregation level is None as fips is unique across aggregation levels.
    df = df.loc[
        (df[CommonFields.POSITIVE_TESTS].notnull())
        & (df[CommonFields.NEGATIVE_TESTS].notnull())
        & (df[CommonFields.NEGATIVE_TESTS] > 0)
        & (df[CommonFields.POSITIVE_TESTS] > 0),
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
    df["expected_positives_from_test_increase"] = pyseir.utils.ewma_smoothing(
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


@lru_cache(maxsize=None)
def load_mobility_data_m50():
    """
    Return mobility data without normalization

    Returns
    -------
    : pd.DataFrame
    """
    return pd.read_pickle(os.path.join(DATA_DIR, "mobility_data__m50.pkl"))


@lru_cache(maxsize=None)
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


@lru_cache(maxsize=None)
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
    path = pyseir.utils.get_run_artifact_path(
        fips=PLACEHOLDER_FIPS, artifact=pyseir.utils.RunArtifact.WHITELIST_RESULT
    )
    return pd.read_json(path, dtype={"fips": str})


def cache_all_data():
    """
    Download all datasets locally.
    """
    cache_mobility_data()
    cache_public_implementations_data()


if __name__ == "__main__":
    cache_all_data()
