import os
import logging
import urllib.request
from typing import Tuple

import requests
import re
import io
import us
import zipfile
import json
from datetime import datetime
from functools import lru_cache

import pandas as pd
import numpy as np

from datapublic.common_fields import CommonFields
from libs.datasets.timeseries import OneRegionTimeseriesDataset
import pyseir.utils


log = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pyseir_data")


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


def calculate_new_case_data_by_region(
    region_timeseries: OneRegionTimeseriesDataset,
    t0: datetime,
    include_testing_correction: bool = False,
    testing_correction_smoothing_tau: float = 5,
) -> Tuple[np.array, np.array]:
    """
    Calculate new cases from combined data.

    Args:
        region_timeseries: Combined data for a region
        t0: Datetime to offset by.
        include_testing_correction: If True, include a correction for new expanded or decreaseed
          test coverage.
        testing_correction_smoothing_tau: expected_positives_from_test_increase is smoothed based
          on an exponentially weighted moving average of decay factor specified here.

    Returns:
        times: List of float days since t0 for the case counts below
        observed_new_cases: Array of integer new cases observed each day.
    """
    assert not region_timeseries.empty
    assert region_timeseries.has_one_region()
    columns = [CommonFields.NEW_CASES]
    county_case_timeseries = region_timeseries.get_subset(
        columns=([CommonFields.LOCATION_ID, CommonFields.DATE] + columns)
    ).remove_padded_nans(columns)
    county_case_data = county_case_timeseries.data

    times_new = (county_case_data[CommonFields.DATE] - t0).dt.days.iloc[1:]

    observed_new_cases = county_case_data[CommonFields.NEW_CASES]
    # Converting to numpy and trimming off the first datapoint to match previous logic.
    # TODO(chris): update logic to be a date indexed series so that this is not necesary.
    observed_new_cases = observed_new_cases.to_numpy()[1:]

    if include_testing_correction:
        df_new_tests = calculate_new_test_data_by_region(
            region_timeseries, t0, smoothing_tau=testing_correction_smoothing_tau
        )
        df_cases = pd.DataFrame({"times": times_new, "new_cases": observed_new_cases})
        df_cases = df_cases.merge(df_new_tests, how="left", on="times")
        df_cases["new_cases"] -= df_cases["expected_positives_from_test_increase"].fillna(0)
        observed_new_cases = df_cases["new_cases"].values

    # Clip because there are sometimes negatives either due to data reporting or
    # corrections in case count. These are always tiny so we just make
    # downstream easier to work with by clipping.
    return times_new, observed_new_cases.clip(min=0)


def calculate_new_test_data_by_region(
    timeseries_dataset: OneRegionTimeseriesDataset,
    t0: datetime,
    smoothing_tau=5,
    correction_threshold=5,
):
    """
    Return a timeseries of new tests for a geography. Note that due to reporting
    discrepancies county to county, and state-to-state, these often do not go
    back as far as case data.

    Args:
        timeseries_dataset: Data for the region
        t0: Reference datetime to use.

    Returns:
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


def cache_all_data():
    """
    Download all datasets locally.
    """
    cache_mobility_data()
    cache_public_implementations_data()


if __name__ == "__main__":
    cache_all_data()
