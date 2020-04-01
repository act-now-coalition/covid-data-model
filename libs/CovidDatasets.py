import datetime
import git
import logging
import math
from copy import copy
import pandas as pd
import os.path
import os
import urllib
from urllib.request import urlopen

import tempfile

from libs.build_params import US_STATE_ABBREV as us_state_abbrev

local_public_data_dir = tempfile.TemporaryDirectory()
local_public_data = local_public_data_dir.name

_logger = logging.getLogger(__name__)

PUBLIC_DATA_REPO = 'https://github.com/covid-projections/covid-data-public'


def get_public_data_base_url():
    # COVID_DATA_PUBLIC could be set, to for instance
    # "https://raw.githubusercontent.com/covid-projections/covid-data-public/master"
    # which would not locally copy the data.

    if not os.getenv('COVID_DATA_PUBLIC', False):
        create_local_copy_public_data()
    return os.getenv('COVID_DATA_PUBLIC')


# TODO: support passing a git hash
def _clone_and_hydrate_repo(repo_url: str, target_dir: str):
    repo = git.Repo.clone_from(repo_url, target_dir)
    # this translates to calling `git lfs fetch` directly on the repo
    # See: https://gitpython.readthedocs.io/en/stable/tutorial.html#using-git-directly
    repo.git.lfs('fetch')


def create_local_copy_public_data():
    """
    Creates a local copy of the public data repository. This is done to avoid
    downloading the file again for each intervention type.
    """
    public_data_local_url = f'file://localhost{local_public_data}/'
    _logger.info(f"Creating a Local Copy of {PUBLIC_DATA_REPO} at {public_data_local_url}")
    _clone_and_hydrate_repo(PUBLIC_DATA_REPO, local_public_data)

    os.environ['COVID_DATA_PUBLIC'] = public_data_local_url


class Dataset:
    """Base class of the Dataset objects. Standardizes the output so that data
    from multiple differenmt sources can be fed into the model with as little
    hassle as possible."""

    @property
    def population_url(self):
        return f"{get_public_data_base_url()}/data/misc/populations.csv"

    @property
    def beds_url(self):
        return f"{get_public_data_base_url()}/data/beds-kff/beds.csv"

    # The output fields need to be standardized, regardless of the input fieldnames. It is the responsibility of the
    #  child class to conform the fieldnames of their data to these fieldnames
    DATE_FIELD = 'date'
    COUNTRY_FIELD = 'country'
    STATE_FIELD = 'state'
    COUNTY_FIELD = 'county'
    CASE_FIELD = 'cases'
    DEATH_FIELD = 'deaths'
    RECOVERED_FIELD = 'recovered'
    SYNTHETIC_FIELD = 'synthetic'

    def __init__(self, start_date, filter_past_date=None):
        if filter_past_date is not None:
            self.filter_past_date = pd.Timestamp(filter_past_date)
        else:
            self.filter_past_date = None

        self._START_DATE = start_date
        self._TIME_SERIES_DATA = None
        self._BED_DATA = None
        self._POPULATION_DATA = None

    def backfill_to_init_date(self, series, model_interval):
        # We need to make sure that the data starts from Mar3, no matter when our records begin
        series = series.sort_values(self.DATE_FIELD).reset_index(drop=True)  # Sort the series by the date of the record
        data_rows = series[series[self.CASE_FIELD] > 0]  # Find those rows that actually have reported cases
        interval_rows = data_rows[data_rows[self.DATE_FIELD].apply(
            lambda d: (d - self._START_DATE).days % model_interval == 0
        )]
        min_interval_row = interval_rows[interval_rows[self.DATE_FIELD] == interval_rows[self.DATE_FIELD].min()].iloc[0]
        series = series[series[self.DATE_FIELD] >= min_interval_row[self.DATE_FIELD]]

        series[self.SYNTHETIC_FIELD] = None  # Create the synthetic record flag field
        # The number of days we need to create to backfill to Mar3
        synthetic_interval = (series[self.DATE_FIELD].min() - self._START_DATE).days
        template = series.iloc[0]  # Grab a row for us to copy structure and data from
        synthetic_data = []
        pd.set_option('mode.chained_assignment', None)  # Suppress pandas' anxiety. We know what we're doing
        for i in range(0, synthetic_interval):
            synthetic_row = template
            synthetic_row[self.DATE_FIELD] = self._START_DATE + datetime.timedelta(days=i)
            synthetic_row[self.CASE_FIELD] = 0
            synthetic_row[self.DEATH_FIELD] = 0
            synthetic_row[self.RECOVERED_FIELD] = 0
            synthetic_row[self.SYNTHETIC_FIELD] = 1
            synthetic_data.append(copy(synthetic_row))  # We need to copy it to prevent alteration by reference
        pd.set_option('mode.chained_assignment', 'warn')  # Turn the anxiety back on
        # Take the synthetic data, and glue it to the bottom of the real records
        return series.append(pd.DataFrame(synthetic_data)).sort_values(self.DATE_FIELD).reset_index(drop=True)

    def step_down(self, i, series, model_interval):
        # A function to calculate how much to step down the number of cases from the following day
        #  The goal is for the synthetic cases to halve once every iteration of the model interval.
        #
        # interval_rows = data_rows[data_rows[self.DATE_FIELD].apply(lambda d: (d - self.start_date).days % model_interval == 0)]
        # min_interval_row = interval_rows[interval_rows[self.DATE_FIELD] == interval_rows[self.DATE_FIELD].min()].iloc[0]

        min_row = series[series[self.CASE_FIELD] > 0].min()
        y = min_row[self.CASE_FIELD] / (math.pow(2, (1 / model_interval)))
        return y

    def backfill_synthetic_cases(self, series, model_interval):
        # Fill in all values prior to the first non-zero values. Use 1/2 following value. Decays into nothing
        #  sort the dataframe in reverse date order, so we traverse from latest to earliest
        for a in range(0, len(series)):
            i = len(series) - a - 1
            if series.iloc[i][self.CASE_FIELD] == 0:
                series.at[i, self.CASE_FIELD] = self.step_down(i, series, model_interval)
        return series

    def backfill(self, series, model_interval):
        # Backfill the data as necessary for the model
        return self.backfill_synthetic_cases(
            self.backfill_to_init_date(series, model_interval),
            model_interval
        )

    def cutoff(self, series):
        # The model MUST start on a certain day. If there is data that precedes that date,
        #  we must trim it from the series
        return series[series[self.DATE_FIELD] >= self._START_DATE]

    def prep_data(self, series, model_interval):
        # We have some requirements of the data's window, and that is enforced here.
        return self.backfill(self.cutoff(series), model_interval)

    def get_all_timeseries(self):
        if self._TIME_SERIES_DATA is None:
            self._TIME_SERIES_DATA = self.get_raw_timeseries()
            if self.filter_past_date is not None:
                self._TIME_SERIES_DATA \
                    = self._TIME_SERIES_DATA[self._TIME_SERIES_DATA[self.DATE_FIELD] <= self.filter_past_date]
        if len(self._TIME_SERIES_DATA.index) == 0:
            raise Exception('No timeseries data entries.')
        return self._TIME_SERIES_DATA

    def get_raw_timeseries(self):
        raise NotImplementedError('The \'get_raw_timeseries\' method must be overriden by the child class')

    def get_all_population(self):
        raise NotImplementedError('The \'get_all_population\' method must be overriden by the child class')

    def get_all_beds(self):
        raise NotImplementedError('The \'get_all_beds\' method must be overriden by the child class')

    def combine_state_county_data(self, country, state):
        # Create a single dataset from state and county data, using state data preferentially.
        # First, pull all available state data
        state_data = self.get_all_timeseries()[
            (self.get_all_timeseries()[self.STATE_FIELD] == state) &
            (self.get_all_timeseries()[self.COUNTRY_FIELD] == country) &
            (self.get_all_timeseries()[self.COUNTY_FIELD].isna())
            ].reset_index(drop=True)
        # Second pull all county data for the state
        county_data = self.get_all_timeseries()[
            (self.get_all_timeseries()[self.STATE_FIELD] == state) &
            (self.get_all_timeseries()[self.COUNTRY_FIELD] == country) &
            (self.get_all_timeseries()[self.COUNTY_FIELD].notna())
            ][[self.DATE_FIELD, self.COUNTRY_FIELD, self.STATE_FIELD, self.CASE_FIELD, self.DEATH_FIELD, self.RECOVERED_FIELD]].groupby(
            [self.DATE_FIELD, self.COUNTRY_FIELD, self.STATE_FIELD], as_index=False
        )[[self.CASE_FIELD, self.DEATH_FIELD, self.RECOVERED_FIELD]].sum()

        # Fill holes.
        state_data = state_data.fillna({'deaths': 0})
        county_data = county_data.fillna({'deaths': 0})

        # Fill holes.
        state_data = state_data.fillna({'deaths': 0})
        county_data = county_data.fillna({'deaths': 0})

        if len(state_data.index) == 0 and len(county_data.index) == 0:
            self.get_all_timeseries()[(self.get_all_timeseries()["state"] == state)].head()
            raise Exception('No county or state-level date for {}, {}'.format(state, country))

        # Now we fill in whatever gaps we can in the state data using the county data
        curr_date = max(state_data[self.DATE_FIELD].max(), county_data[self.DATE_FIELD].max()) # Start on the last date of state data we have
        county_data_to_insert = []
        while curr_date > self._START_DATE:
            # If there is no state data for a day, we need to get some country data for the day
            if len(state_data[state_data[self.DATE_FIELD] == curr_date]) == 0:
                county_data_for_date = copy(county_data[county_data[self.DATE_FIELD] == curr_date])
                if len(county_data_for_date) > 0:
                    county_data_for_date = county_data_for_date.iloc[0]
                    new_state_row = copy(state_data.iloc[0])  # Copy the first row of the state data to get the right format
                    new_state_row[self.DATE_FIELD] = county_data_for_date[self.DATE_FIELD]
                    new_state_row[self.CASE_FIELD] = county_data_for_date[self.CASE_FIELD]
                    new_state_row[self.DEATH_FIELD] = county_data_for_date[self.DEATH_FIELD]
                    new_state_row[self.RECOVERED_FIELD] = county_data_for_date[self.RECOVERED_FIELD]
                    county_data_to_insert.append(copy(new_state_row))
                else:
                    # If there's no county data, we're SOL.
                    _logger.info("NO COUNTY DATA: {}".format(curr_date))
            curr_date -= datetime.timedelta(days=1)
        return state_data.append(pd.DataFrame(county_data_to_insert)).sort_values(self.DATE_FIELD)

    def get_timeseries_by_country_state(self, country, state, model_interval):
        #  Prepare a state-level dataset that uses county data to fill in any potential gaps
        state_data = self.combine_state_county_data(country, state)
        return self.prep_data(state_data, model_interval)

    def get_timeseries_by_country(self, country):
        return self.get_all_timeseries()[self.get_all_timeseries()[self.COUNTRY_FIELD] == country]

    def get_population_by_country_state(self, country, state):
        matching_pops = self.get_all_population()[
            (self.get_all_population()[self.STATE_FIELD] == state) &
            (self.get_all_population()[self.COUNTRY_FIELD] == country)
        ]
        try:
            return int(matching_pops.iloc[0].at["population"])
        except IndexError as e:
            _logger.error('No population data for {}, {}'.format(state, country))
            raise e

    def get_beds_by_country_state(self, country, state):
        matching_beds = self.get_all_beds()[(self.get_all_beds()[self.STATE_FIELD] == state) &
                                  (self.get_all_beds()[self.COUNTRY_FIELD] == country)]
        return int(round(float(matching_beds.iloc[0].at["bedspermille"]) * self.get_population_by_country_state(country, state) / 1000))


class JHUDataset(Dataset):
    # The date of the first JHU data snapshot.
    _FIRST_JHU_DATE = datetime.date(2020, 1, 22)

    @property
    def jhu_url_template(self):
        return f'{get_public_data_base_url()}/data/cases-jhu/csse_covid_19_daily_reports/{{}}.csv'

    def __init__(self, filter_past_date=None):
        start_date = datetime.datetime(year=2020, month=3, day=3)
        super().__init__(start_date=start_date, filter_past_date=filter_past_date)
        self._fieldname_map = {
            'Country/Region': self.COUNTRY_FIELD,
            'Country_Region': self.COUNTRY_FIELD,
            'Province/State': self.STATE_FIELD,
            'Province_State': self.STATE_FIELD,
            'Admin2': self.COUNTY_FIELD,
            'Confirmed': self.CASE_FIELD,
            'Deaths': self.DEATH_FIELD,
            'Recovered': self.RECOVERED_FIELD
        }

    def transform_jhu_timeseries(self):
        """"Takes a list of JHU daily reports, mashes them into a single report, then restructures and renames the data
        to fit the model's expectations"""

        # Compile a list of all of the day reports available
        def parse_county(state):
            if ',' in state:
                return state.split(',')[0].strip()

        def parse_state(state):
            state = state.strip()
            if ',' in state:
                state = state.split(',')[1].strip()
            if state in us_state_abbrev:
                return us_state_abbrev[state]
            return state

        def parse_country(country):
            if country == 'US':
                return 'USA'
            return country

        day_reports = []
        snapshot_date = self._FIRST_JHU_DATE
        while True:
            csv_url = self.jhu_url_template.format(snapshot_date.strftime('%m-%d-%Y'))
            _logger.debug('Loading URL: {}'.format(csv_url))

            # For each data file
            try:
                df = pd.read_csv(csv_url)
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    _logger.info('Received a 404 for date {}. Ending iteration.'.format(snapshot_date))
                    break
                raise
            except urllib.error.URLError as e:
                if isinstance(e.reason, FileNotFoundError):
                    _logger.info('Received a 404 for date {}. Ending iteration.'.format(snapshot_date))
                    break
                raise
            except FileNotFoundError:
                # assuming we're pointing to a locally cached repository
                _logger.info('File not found for date {}. Ending iteration.'.format(snapshot_date))
                break
                raise
                
            df = df.rename(columns=self._fieldname_map)
            snapshot_date_as_datetime = datetime.datetime.combine(snapshot_date, datetime.datetime.min.time())
            df = df.assign(**{self.DATE_FIELD: snapshot_date_as_datetime})
            # Select out the subset of fields we care about
            if self.COUNTY_FIELD not in df.columns:
                df[self.COUNTY_FIELD] = df[self.STATE_FIELD].dropna().apply(parse_county)
            df = df[[self.DATE_FIELD, self.COUNTRY_FIELD, self.STATE_FIELD, self.COUNTY_FIELD, self.CASE_FIELD,
                     self.DEATH_FIELD, self.RECOVERED_FIELD]]
            # Parse the states from their longhand into their abbreviations
            df[self.STATE_FIELD] = df[self.STATE_FIELD].dropna().apply(parse_state)
            # Parse the 'US' entries into 'USA'
            df[self.COUNTRY_FIELD] = df[self.COUNTRY_FIELD].apply(parse_country)
            # Append the record dates by converting the file name into a datetime object
            # Only process the csv files
            day_reports.append(df)
            snapshot_date += datetime.timedelta(days=1)
        full_report = pd.concat(day_reports)  # Concat said reports into a single DataFrame
        full_report = full_report.reset_index()

        full_report.info()
        return full_report.drop('index', axis=1)

    def get_raw_timeseries(self):
        return self.transform_jhu_timeseries()

    def get_all_population(self):
        if(self._POPULATION_DATA is None):
            self._POPULATION_DATA = pd.read_csv(self.population_url)
        return self._POPULATION_DATA

    def get_all_beds(self):
        if(self._BED_DATA is None):
            self._BED_DATA = pd.read_csv(self.beds_url)
        return self._BED_DATA

    def get_all_states_by_country(self, country):
        return self.get_all_population()[self.get_all_population()[self.COUNTRY_FIELD] == country][self.STATE_FIELD].dropna().unique()


class CDSDataset(Dataset):
    """CoronaDataScraper Dataset"""
    @property
    def timeseries_url(self):
        return f"{get_public_data_base_url()}/data/cases-cds/timeseries.csv"

    def __init__(self, filter_past_date=None):
        super().__init__(
            start_date=datetime.datetime(year=2020, month=3, day=3),
            filter_past_date=filter_past_date
        )

    def get_raw_timeseries(self):
        if(self._TIME_SERIES_DATA is None):
            self._TIME_SERIES_DATA = pd.read_csv(self.timeseries_url, parse_dates=[self.DATE_FIELD])
        return self._TIME_SERIES_DATA

    def get_all_population(self):
        if(self._POPULATION_DATA is None):
            self._POPULATION_DATA = pd.read_csv(self.population_url)
        return self._POPULATION_DATA

    def get_all_beds(self):
        if(self._BED_DATA is None):
            self._BED_DATA = pd.read_csv(self.beds_url)
        return self._BED_DATA

    def get_all_states_by_country(self, country):
        return self.get_all_population()[self.get_all_population()[self.COUNTRY_FIELD] == country][self.STATE_FIELD].dropna().unique()
