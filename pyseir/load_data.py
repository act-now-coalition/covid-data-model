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
from libs.datasets.timeseries import TimeseriesDataset
from libs.datasets.dataset_utils import AggregationLevel
from libs.datasets import CovidTrackingDataSource
from pyseir.utils import get_run_artifact_path, RunArtifact
from functools import lru_cache
from enum import Enum


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'pyseir_data')


class HospitalizationDataType(Enum):
    CUMULATIVE_HOSPITALIZATIONS = 'cumulative_hospitalizations'
    CURRENT_HOSPITALIZATIONS = 'current_hospitalizations'


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
        x0 = np.median(input_series[(i - window_size):(i + window_size)])
        S0 = k * np.median(
            np.abs(input_series[(i - window_size):(i + window_size)] - x0))
        if (-(input_series[i] - x0) > n_sigmas * S0):
            new_series[i] = x0
            indices.append(i)

    return new_series, indices


def load_zip_get_file(url, file, decoder='utf-8'):
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
    logging.info('Downloading covid case data')
    # NYT dataset
    county_case_data = pd.read_csv('https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv', dtype='str')
    county_case_data['date'] = pd.to_datetime(county_case_data['date'])
    county_case_data[['cases', 'deaths']] = county_case_data[['cases', 'deaths']].astype(int)
    county_case_data = county_case_data[county_case_data['fips'].notnull()]
    county_case_data.to_pickle(os.path.join(DATA_DIR, 'covid_case_timeseries.pkl'))


def cache_mobility_data():
    """
    Pulled from https://github.com/descarteslabs/DL-COVID-19
    """
    logging.info('Downloading mobility data.')
    url = 'https://raw.githubusercontent.com/descarteslabs/DL-COVID-19/master/DL-us-mobility-daterow.csv'

    dtypes_mapping = {
        'country_code': str,
        'admin_level': int,
        'admin1': str,
        'admin2': str,
        'fips': str,
        'samples': int,
        'm50': float,
        'm50_index': float}

    df = pd.read_csv(filepath_or_buffer=url, parse_dates=['date'], dtype=dtypes_mapping)
    df__m50 = df.query('admin_level == 2')[['fips', 'date', 'm50']]
    df__m50_index = df.query('admin_level == 2')[['fips', 'date', 'm50_index']]
    df__m50__final = df__m50.groupby('fips').agg(list).reset_index()
    df__m50_index__final = df__m50_index.groupby('fips').agg(list).reset_index()
    df__m50__final['m50'] = df__m50__final['m50'].apply(lambda x: np.array(x))
    df__m50_index__final['m50_index'] = df__m50_index__final['m50_index'].apply(lambda x: np.array(x))

    df__m50__final.to_pickle(os.path.join(DATA_DIR, 'mobility_data__m50.pkl'))
    df__m50_index__final.to_pickle(os.path.join(DATA_DIR, 'mobility_data__m50_index.pkl'))


def cache_public_implementations_data():
    """
    Pulled from https://github.com/JieYingWu/COVID-19_US_County-level_Summaries
    """
    logging.info('Downloading public implementations data')
    url = 'https://raw.githubusercontent.com/JieYingWu/COVID-19_US_County-level_Summaries/master/raw_data/national/public_implementations_fips.csv'

    data = requests.get(url, verify=False).content.decode('utf-8')
    data = re.sub(r',(\d+)-(\w+)', r',\1-\2-2020', data)  # NOTE: This assumes the year 2020

    date_cols = [
        'stay at home',
        '>50 gatherings',
        '>500 gatherings',
        'public schools',
        'restaurant dine-in',
        'entertainment/gym',
        'Federal guidelines',
        'foreign travel ban']
    df = pd.read_csv(io.StringIO(data), parse_dates=date_cols, dtype='str').drop(['Unnamed: 1', 'Unnamed: 2'], axis=1)
    df.columns = [col.replace('>', '').replace(' ', '_').replace('/', '_').lower() for col in df.columns]
    df.fips = df.fips.apply(lambda x: x.zfill(5))
    df.to_pickle(os.path.join(DATA_DIR, 'public_implementations_data.pkl'))


@lru_cache(maxsize=32)
def load_county_case_data():
    """
    Return county level case data.

    Returns
    -------
    : pd.DataFrame
    """
    county_case_data = NYTimesDataset.load().timeseries() \
                         .get_subset(AggregationLevel.COUNTY, country='USA') \
                         .get_data(country='USA')
    return county_case_data


@lru_cache(maxsize=1)
def load_state_case_data():
    """
    Return county level case data.

    Returns
    -------
    : pd.DataFrame
    """

    state_case_data = NYTimesDataset.load().timeseries() \
                         .get_subset(AggregationLevel.STATE, country='USA') \
                         .get_data(country='USA')
    return state_case_data


@lru_cache(maxsize=32)
def load_county_metadata():
    """
    Return county level metadata such as age distributions, populations etc..

    Returns
    -------
    : pd.DataFrame

    """

    county_metadata = pd.read_json(os.path.join(DATA_DIR, 'county_metadata.json'), dtype={'fips': 'str'})
    # Fix state names
    county_metadata.loc[:, 'state'] = county_metadata['fips'].apply(lambda x: us.states.lookup(x[:2]).name)
    return county_metadata


@lru_cache(maxsize=32)
def load_county_metadata_by_state(state):
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

    if state:
        state = [state] if not isinstance(state, list) else state
        state_metadata = state_metadata[state_metadata.state.isin(state)]

    density_measures = ['housing_density', 'population_density']
    for col in density_measures:
        state_metadata.loc[:, col] = state_metadata[col] * state_metadata['total_population']

    age_dist = state_metadata.groupby('state')['age_distribution'] \
                             .apply(lambda l: np.stack(np.array(l)).sum(axis=0))
    density_info = state_metadata.groupby('state').agg(
        {'population_density': lambda x: sum(x),
         'housing_density': lambda x: sum(x),
         'total_population': lambda x: sum(x),
         'fips': list})
    age_bins = state_metadata[['state', 'age_bin_edges']].groupby('state').first()
    state_metadata = pd.concat([age_dist, density_info, age_bins], axis=1)

    for col in density_measures:
        state_metadata[col] /= state_metadata['total_population']

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
    with open(output_filename) as f:
        fit_results = json.load(f)
    return fit_results


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
    county_metadata_merged = county_metadata.set_index('fips').loc[fips].to_dict()
    for key, value in county_metadata_merged.items():
        if np.isscalar(value) and not isinstance(value, str):
            county_metadata_merged[key] = float(value)
    return county_metadata_merged


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
    county_case_data = _county_case_data[_county_case_data['fips'] == fips]
    times_new = (county_case_data['date'] - t0).dt.days.iloc[1:]
    observed_new_cases = county_case_data['cases'].values[1:] - county_case_data['cases'].values[:-1]
    observed_new_deaths = county_case_data['deaths'].values[1:] - county_case_data['deaths'].values[:-1]

    # Clip because there are sometimes negatives either due to data reporting or
    # corrections in case count. These are always tiny so we just make
    # downstream easier to work with by clipping.
    return times_new, observed_new_cases.clip(min=0), observed_new_deaths.clip(min=0)


def get_hospitalization_data():
    data = CovidTrackingDataSource.local().timeseries(fill_na=False).data
    # Since we're using this data for hospitalized data only, only returning
    # values with hospitalization data.  I think as the use cases of this data source
    # expand, we may not want to drop. For context, as of 4/8 607/1821 rows contained
    # hospitalization data.
    has_current_hospital = data[TimeseriesDataset.Fields.CURRENT_HOSPITALIZED].notnull()
    has_cumulative_hospital = data[TimeseriesDataset.Fields.CUMULATIVE_HOSPITALIZED].notnull()
    return TimeseriesDataset(data[has_current_hospital | has_cumulative_hospital])


@lru_cache(maxsize=32)
def load_hospitalization_data(fips, t0, category='hospitalized'):
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
    hospitalization_data = get_hospitalization_data()\
        .get_subset(AggregationLevel.COUNTY, country='USA', fips=fips) \
        .get_data(country='USA', fips=fips)

    if len(hospitalization_data) == 0:
        return None, None, None

    if (hospitalization_data[f'current_{category}'] > 0).any():
        hospitalization_data = hospitalization_data[hospitalization_data[f'current_{category}'].notnull()]
        relative_days = (hospitalization_data['date'].dt.date - t0.date()).dt.days.values
        return relative_days, \
               hospitalization_data[f'current_{category}'].values.clip(min=0),\
               HospitalizationDataType.CURRENT_HOSPITALIZATIONS
    elif (hospitalization_data[f'cumulative_{category}'] > 0).any():
        hospitalization_data = hospitalization_data[hospitalization_data[f'cumulative_{category}'].notnull()]
        relative_days = (hospitalization_data['date'].dt.date - t0.date()).dt.days.values
        cumulative = hospitalization_data[f'cumulative_{category}'].values.clip(min=0)
        # Some minor glitches for a few states..
        for i, val in enumerate(cumulative[1:]):
            if cumulative[i] > cumulative[i+1]:
                cumulative[i] = cumulative[i + 1]
        return relative_days, cumulative, HospitalizationDataType.CUMULATIVE_HOSPITALIZATIONS
    else:
        return None, None, None


@lru_cache(maxsize=32)
def load_hospitalization_data_by_state(state, t0, convert_cumulative_to_current=False, category='hospitalized'):
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
    convert_cumulative_to_current: bool
        If True, and only cumulative hospitalizations are available, convert the
        current hospitalizations to the current value.
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
    hospitalization_data = (
        CovidTrackingDataSource.local().timeseries(fill_na=False)
        .get_subset(AggregationLevel.STATE, country='USA', state=abbr)
        .get_data(country='USA', state=abbr)
    )

    categories = ['icu', 'hospitalized']
    if category not in categories:
        raise ValueError(f'Hospitalization category {category} is not in {categories}')

    if len(hospitalization_data) == 0:
        return None, None, None

    if (hospitalization_data[f'current_{category}'] > 0).any():
        hospitalization_data = hospitalization_data[hospitalization_data[f'current_{category}'].notnull()]
        times_new = (hospitalization_data['date'].dt.date - t0.date()).dt.days.values
        return times_new, \
               hospitalization_data[f'current_{category}'].values.clip(min=0), \
               HospitalizationDataType.CURRENT_HOSPITALIZATIONS
    elif (hospitalization_data[f'cumulative_{category}'] > 0).any():
        hospitalization_data = hospitalization_data[hospitalization_data[f'cumulative_{category}'].notnull()]
        times_new = (hospitalization_data['date'].dt.date - t0.date()).dt.days.values
        cumulative = hospitalization_data[f'cumulative_{category}'].values.clip(min=0)
        # Some minor glitches for a few states..
        for i, val in enumerate(cumulative[1:]):
            if cumulative[i] > cumulative[i + 1]:
                cumulative[i] = cumulative[i + 1]

        if convert_cumulative_to_current:
            # Must be here to avoid circular import. This is required to convert
            # cumulative hosps to current hosps. We also just use a dummy fips and t_list.
            from pyseir.parameters.parameter_ensemble_generator import ParameterEnsembleGenerator
            params = ParameterEnsembleGenerator(fips='06', t_list=[], N_samples=1).get_average_seir_parameters()
            if category == 'hospitalized':
                average_length_of_stay = (
                      params['hospitalization_rate_general'] * params['hospitalization_length_of_stay_general']
                    + params['hospitalization_rate_icu'] * (1 - params['fraction_icu_requiring_ventilator']) * params['hospitalization_length_of_stay_icu']
                    + params['hospitalization_rate_icu'] * params['fraction_icu_requiring_ventilator'] * params['hospitalization_length_of_stay_icu_and_ventilator']
                ) / (params['hospitalization_rate_general'] + params['hospitalization_rate_icu'])
            else:
                average_length_of_stay = (
                    (1 - params['fraction_icu_requiring_ventilator']) * params['hospitalization_length_of_stay_icu']
                    + params['fraction_icu_requiring_ventilator'] * params['hospitalization_length_of_stay_icu_and_ventilator'])

            # Now compute a cumulative sum, but at each day, subtract the discharges from the previous count.
            new_hospitalizations = np.append([0], np.diff(cumulative))
            current = [0]
            for i, new_hosps in enumerate(new_hospitalizations[1:]):
                current.append(current[i] + new_hosps - current[i] / average_length_of_stay)
            return times_new, current, HospitalizationDataType.CURRENT_HOSPITALIZATIONS
        else:
            return times_new, cumulative, HospitalizationDataType.CUMULATIVE_HOSPITALIZATIONS
    else:
        return None, None, None


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
    state_case_data = _state_case_data[_state_case_data['state'] == us.states.lookup(state).abbr]
    times_new = (state_case_data['date'] - t0).dt.days.iloc[1:]
    observed_new_cases = state_case_data['cases'].values[1:] - state_case_data['cases'].values[:-1]
    observed_new_deaths = state_case_data['deaths'].values[1:] - state_case_data['deaths'].values[:-1]

    _, filter_idx = hampel_filter__low_outliers_only(observed_new_cases, window_size=5, n_sigmas=2)
    keep_idx = np.array([i for i in range(len(times_new)) if i not in list(filter_idx)])
    times_new = [int(list(times_new)[idx]) for idx in keep_idx]
    return times_new, np.array(observed_new_cases[keep_idx]).clip(min=0), observed_new_deaths.clip(min=0)[keep_idx]


@lru_cache(maxsize=1)
def load_mobility_data_m50():
    """
    Return mobility data without normalization

    Returns
    -------
    : pd.DataFrame
    """
    return pd.read_pickle(os.path.join(DATA_DIR, 'mobility_data__m50.pkl'))


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
    return pd.read_pickle(os.path.join(DATA_DIR, 'mobility_data__m50_index.pkl')).set_index('fips')


@lru_cache(maxsize=1)
def load_public_implementations_data():
    """
    Return public implementations data

    Returns
    -------
    : pd.DataFrame
    """
    return pd.read_pickle(os.path.join(DATA_DIR, 'public_implementations_data.pkl')).set_index('fips')


def load_whitelist():
    """
    Load the whitelist result.

    Returns
    -------
    whitelist: pd.DataFrame
        DataFrame containing a whitelist of product features for counties.
    """
    path = get_run_artifact_path(
        fips='06', # dummy since not used for whitelist.
        artifact=RunArtifact.WHITELIST_RESULT)
    return pd.read_json(path, dtype={'fips': str})


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
    simulation_start_date = datetime.fromisoformat(load_inference_result(fips)['t0_date'])
    date_idx = int((date - simulation_start_date).days)
    return ensemble_results['suppression_policy__inferred'][compartment]['ci_50'][date_idx]


if __name__ == '__main__':
    cache_all_data()
