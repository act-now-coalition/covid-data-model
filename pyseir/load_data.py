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


def load_county_case_data():
    """
    Return county level case data. The following columns:

    Returns
    -------
    : pd.DataFrame
    """
    return pd.read_pickle(os.path.join(DATA_DIR, 'covid_case_timeseries.pkl'))


def load_county_metadata():
    """
    Return county level metadata such as age distributions, populations etc..

    Returns
    -------
    : pd.DataFrame

    """
    return pd.read_json(os.path.join(DATA_DIR, 'county_metadata.json'), dtype={'fips': 'str'})


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
        path = os.path.join(OUTPUT_DIR, state, 'data', f"{state}__{county}__{fips}__{run_mode}__ensemble_projections.json")
    elif len(fips) == 2:
        state = us.states.lookup(fips).name
        path = os.path.join(OUTPUT_DIR, state, 'data', f"{state}__{fips}__{run_mode}__ensemble_projections.json")

    with open(path) as f:
        fit_results = json.load(f)
    return fit_results


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
    return times_new, observed_new_cases, observed_new_deaths


def load_hospital_data():
    """
    Return hospital level data. Note that this must be aggregated by stcountyfp
    to obtain county level estimates.

    Returns
    -------
    : pd.DataFrame
    """
    return pd.read_pickle(os.path.join(DATA_DIR, 'icu_capacity.pkl'))


def load_mobility_data_m50():
    """
    Return mobility data without normalization

    Returns
    -------
    : pd.DataFrame
    """
    return pd.read_pickle(os.path.join(DATA_DIR, 'mobility_data__m50.pkl'))


# Ensembles need to access this 1e6 times and it makes 10ms simulations -> 100 ms otherwise.
in_memory_cache = None
def load_mobility_data_m50_index():
    """
    Return mobility data with normalization: per
    https://github.com/descarteslabs/DL-COVID-19 normal m50 is defined during
    2020-02-17 to 2020-03-07.

    Returns
    -------
    : pd.DataFrame
    """
    global in_memory_cache
    if in_memory_cache is not None:
        return in_memory_cache
    else:
        in_memory_cache = pd.read_pickle(os.path.join(DATA_DIR, 'mobility_data__m50_index.pkl')).set_index('fips')

    return in_memory_cache.copy()


def load_public_implementations_data():
    """
    Return public implementations data

    Returns
    -------
    : pd.DataFrame
    """
    return pd.read_pickle(os.path.join(DATA_DIR, 'public_implementations_data.pkl'))


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
