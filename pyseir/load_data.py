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
from pyseir import OUTPUT_DIR
from libs.datasets import NYTimesDataset
from libs.datasets.dataset_utils import AggregationLevel
from libs.datasets import CovidTrackingDataSource
from functools import lru_cache


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'pyseir_data')


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


def cache_hospital_beds():
    """
    Pulled from "Definitive"
    See: https://services7.arcgis.com/LXCny1HyhQCUSueu/arcgis/rest/services/Definitive_Healthcare_Hospitals_Beds_Hospitals_Only/FeatureServer/0
    """
    logging.info('Downloading ICU capacity data.')
    url = 'http://opendata.arcgis.com/datasets/f3f76281647f4fbb8a0d20ef13b650ca_0.geojson'
    tmp_file = urllib.request.urlretrieve(url)[0]

    with open(tmp_file) as f:
        vals = json.load(f)
    df = pd.DataFrame([val['properties'] for val in vals['features']])
    df.columns = [col.lower() for col in df.columns]
    df = df.drop(['objectid', 'state_fips', 'cnty_fips'], axis=1)
    df.to_pickle(os.path.join(DATA_DIR, 'icu_capacity.pkl'))


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
    county_metadata.loc[:, 'state'] = county_metadata['fips'].apply(lambda x: us.states.lookup(x[:2]).name).str.title()
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
def load_ensemble_results(fips, run_mode='default'):
    """
    Retrieve ensemble results for a given state or county fips code.

    Parameters
    ----------
    fips: str
        County FIPS to load.
    run_mode: str
        Which run mode to pull results from.

    Returns
    -------
    ensemble_results: dict
    """
    if len(fips) == 5:  # County
        county_metadata = load_county_metadata().set_index('fips')
        state, county = county_metadata.loc[fips]['state'], county_metadata.loc[fips]['county']
        path = os.path.join(OUTPUT_DIR, 'pyseir', state, 'data', f"{state}__{county}__{fips}__{run_mode}__ensemble_projections.json")
    elif len(fips) == 2:
        state = us.states.lookup(fips).name
        path = os.path.join(OUTPUT_DIR, 'pyseir', state, 'data', f"{state}__{fips}__{run_mode}__ensemble_projections.json")

    with open(path) as f:
        fit_results = json.load(f)
    return fit_results


@lru_cache(maxsize=32)
def load_county_metadata_by_fips(fips):
    """
    Generate a dictionary for a county which includes county metadata merged
    with hospital capacity data.

    Parameters
    ----------
    fips: str

    Returns
    -------
    county_metadata: dict
        Dictionary of metadata for the county. The keys are:

        ['state', 'county', 'total_population', 'population_density',
        'housing_density', 'age_distribution', 'age_bin_edges',
        'num_licensed_beds', 'num_staffed_beds', 'num_icu_beds',
        'bed_utilization', 'potential_increase_in_bed_capac']
    """
    county_metadata = load_county_metadata()
    hospital_bed_data = load_hospital_data()

    # Not all counties have hospital data.
    hospital_bed_data = hospital_bed_data[
        ['fips',
         'num_licensed_beds',
         'num_staffed_beds',
         'num_icu_beds',
         'bed_utilization',
         'potential_increase_in_bed_capac']].groupby('fips').sum()

    county_metadata_merged = county_metadata.merge(hospital_bed_data, on='fips', how='left').set_index('fips').loc[fips].to_dict()
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

    return times_new, observed_new_cases.clip(min=0), observed_new_deaths.clip(min=0)

@lru_cache(maxsize=32)
def load_hospitalization_data(fips, t0):
    """
    Obtain hospitalization data.

    Parameters
    ----------
    fips: str
        County fips to lookup.
    t0: datetime
        Datetime to offset by.

    Returns
    -------
    times: array(float)
        List of float days since t0 for the hospitalization data.
    observed_hospitalizations: array(int)
        Array of new cases observed each day.
    type: str
        'cumulative' or 'current'
    """
    hospitalization_data = CovidTrackingDataSource.local().timeseries()\
        .get_subset(AggregationLevel.COUNTY, country='USA', fips=fips) \
        .get_data(country='USA', fips=fips)

    if len(hospitalization_data) == 0:
        return None, None, None

    times_new = (hospitalization_data['date'].dt.date - t0.date()).dt.days.values

    if (hospitalization_data['current_hospitalized'] > 0).any():
        return times_new, hospitalization_data['current_hospitalized'].values.clip(min=0), 'current'
    elif (hospitalization_data['cumulative_hospitalized'] > 0).any():
        return times_new, hospitalization_data['cumulative_hospitalized'].values.clip(min=0), 'cumulative'
    else:
        return None, None, None


@lru_cache(maxsize=32)
def load_hospitalization_data_by_state(state, t0):
    """
    Obtain hospitalization data.

    Parameters
    ----------
    state: str
        State to lookup.
    t0: datetime
        Datetime to offset by.

    Returns
    -------
    times: array(float) or NoneType
        List of float days since t0 for the hospitalization data.
    observed_hospitalizations: array(int) or NoneType
        Array of new cases observed each day.
    type: str
        'cumulative' or 'current' or NoneType
    """
    abbr = us.states.lookup(state).abbr
    hospitalization_data = CovidTrackingDataSource.local().timeseries()\
        .get_subset(AggregationLevel.STATE, country='USA', state=abbr) \
        .get_data(country='USA', state=abbr)

    if len(hospitalization_data) == 0:
        return None, None, None

    times_new = (hospitalization_data['date'].dt.date - t0.date()).dt.days.values

    if (hospitalization_data['current_hospitalized'] > 0).any():
        return times_new, hospitalization_data['current_hospitalized'].values.clip(min=0), 'current'
    elif (hospitalization_data['cumulative_hospitalized'] > 0).any():
        return times_new, hospitalization_data['cumulative_hospitalized'].values.clip(min=0), 'cumulative'
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

    return times_new, observed_new_cases.clip(min=0), observed_new_deaths.clip(min=0)


def load_hospital_data():
    """
    Return hospital level data. Note that this must be aggregated by stcountyfp
    to obtain county level estimates.

    Returns
    -------
    : pd.DataFrame
    """
    return pd.read_pickle(os.path.join(DATA_DIR, 'icu_capacity.pkl'))


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


def cache_all_data():
    """
    Download all datasets locally.
    """
    cache_county_case_data()
    cache_hospital_beds()
    cache_mobility_data()
    cache_public_implementations_data()


if __name__ == '__main__':
    cache_all_data()
