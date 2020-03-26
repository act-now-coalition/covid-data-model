import datetime
import logging
import math
from copy import copy
import pandas as pd
import os.path
import datetime

# Dict to transform longhand state names to abbreviations
us_state_abbrev = {
    'Alabama': 'AL',
    'Alaska': 'AK',
    'American Samoa': 'AS',
    'Arizona': 'AZ',
    'Arkansas': 'AR',
    'California': 'CA',
    'Colorado': 'CO',
    'Connecticut': 'CT',
    'Delaware': 'DE',
    'District of Columbia': 'DC',
    'Florida': 'FL',
    'Georgia': 'GA',
    'Guam': 'GU',
    'Hawaii': 'HI',
    'Idaho': 'ID',
    'Illinois': 'IL',
    'Indiana': 'IN',
    'Iowa': 'IA',
    'Kansas': 'KS',
    'Kentucky': 'KY',
    'Louisiana': 'LA',
    'Maine': 'ME',
    'Maryland': 'MD',
    'Massachusetts': 'MA',
    'Michigan': 'MI',
    'Minnesota': 'MN',
    'Mississippi': 'MS',
    'Missouri': 'MO',
    'Montana': 'MT',
    'Nebraska': 'NE',
    'Nevada': 'NV',
    'New Hampshire': 'NH',
    'New Jersey': 'NJ',
    'New Mexico': 'NM',
    'New York': 'NY',
    'North Carolina': 'NC',
    'North Dakota': 'ND',
    'Northern Mariana Islands':'MP',
    'Ohio': 'OH',
    'Oklahoma': 'OK',
    'Oregon': 'OR',
    'Pennsylvania': 'PA',
    'Puerto Rico': 'PR',
    'Rhode Island': 'RI',
    'South Carolina': 'SC',
    'South Dakota': 'SD',
    'Tennessee': 'TN',
    'Texas': 'TX',
    'Utah': 'UT',
    'Vermont': 'VT',
    'Virgin Islands': 'VI',
    'Virginia': 'VA',
    'Washington': 'WA',
    'West Virginia': 'WV',
    'Wisconsin': 'WI',
    'Wyoming': 'WY'
}

class Dataset:
    """Base class of the Dataset objects. Standardizes the output so that data from multiple differet sources can 
    be fed into the model with as little hassle as possible."""
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
            synthetic_data.append(copy(synthetic_row)) # We need to copy it to prevent alteration by reference
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
        # Now we fill in whatever gaps we can in the state data using the county data
        curr_date = state_data[self.DATE_FIELD].max()  # Start on the last date of state data we have
        county_data_to_insert = []
        while curr_date > self._START_DATE:
            curr_date -= datetime.timedelta(days=1)
            # If there is no state data for a day, we need to get some country data for the day
            if len(state_data[state_data[self.DATE_FIELD] == curr_date]) == 0:
                county_data_for_date = copy(county_data[county_data[self.DATE_FIELD] == curr_date])
                if len(county_data_for_date) == 0:  # If there's no county data, we're SOL.
                    continue  # TODO: Revisit. This should be more intelligent
                county_data_for_date = county_data_for_date.iloc[0]
                new_state_row = copy(state_data.iloc[0])  # Copy the first row of the state data to get the right format
                new_state_row[self.DATE_FIELD] = county_data_for_date[self.DATE_FIELD]
                new_state_row[self.CASE_FIELD] = county_data_for_date[self.CASE_FIELD]
                new_state_row[self.DEATH_FIELD] = county_data_for_date[self.DEATH_FIELD]
                new_state_row[self.RECOVERED_FIELD] = county_data_for_date[self.RECOVERED_FIELD]
                county_data_to_insert.append(copy(new_state_row))
        return state_data.append(pd.DataFrame(county_data_to_insert)).sort_values(self.DATE_FIELD)

    def get_timeseries_by_country_state(self, country, state, model_interval):
        #  Prepare a state-level dataset that uses county data to fill in any potential gaps
        return self.prep_data(self.combine_state_county_data(country, state), model_interval)

    def get_timeseries_by_country(self, country):
        return self.get_all_timeseries()[self.get_all_timeseries()[self.COUNTRY_FIELD] == country]

    def get_population_by_country_state(self, country, state):
        matching_pops = self.get_all_population()[(self.get_all_population()[self.STATE_FIELD] == state) & (
        self.get_all_population()[self.COUNTRY_FIELD] == country)]
        try:
            return int(matching_pops.iloc[0].at["population"])
        except IndexError as e:
            logging.error('No population data for {}, {}'.format(state, country))
            raise e

    def get_beds_by_country_state(self, country, state):
        matching_beds = self.get_all_beds()[(self.get_all_beds()[self.STATE_FIELD] == state) &
                                  (self.get_all_beds()[self.COUNTRY_FIELD] == country)]
        return int(round(float(matching_beds.iloc[0].at["bedspermille"]) * self.get_population_by_country_state(country, state) / 1000))


class JHUDataset(Dataset):
    _CONFIRMED_GLOBAL_URL = r'https://github.com/CSSEGISandData/COVID-19/blob/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
    _DEATHS_GLOBAL_URL = r'https://github.com/CSSEGISandData/COVID-19/blob/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'
    _POPULATION_URL = r'https://raw.githubusercontent.com/covid-projections/covid-data-model/master/data/populations.csv'
    _BEDS_URL = r'https://raw.githubusercontent.com/covid-projections/covid-data-model/master/data/beds.csv'

    # JHU seems to change the names of their fields a lot, so this provides a simple way to wrangle them all


    def __init__(self, filter_past_date=None):
        super().__init__(start_date=datetime.datetime(year=2020, month=3, day=3), filter_past_date=filter_past_date)
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
        daily_reports_dir = os.path.join('data', 'jhu', 'csse_covid_19_daily_reports')
        # Compile a list of all of the day reports available
        def parse_county(state):
            if ',' in state:
                return state.split(',')[0]

        def parse_state(state):
            if ',' in state:
                state = state.split(',')[1]
            if state in us_state_abbrev:
                return us_state_abbrev[state]
            return state

        def parse_country(country):
            if country == 'US':
                return 'USA'
            return country

        day_reports = []
        for f in os.listdir(daily_reports_dir):
            if os.path.splitext(f)[1] == '.csv':
                # For each data file in the directory
                df = pd.read_csv(os.path.join(daily_reports_dir, f))
                df = df.rename(columns=self._fieldname_map)
                df = df.assign(**{self.DATE_FIELD: datetime.datetime.strptime(f.split('.')[0], '%m-%d-%Y')})
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
        return pd.concat(day_reports).reset_index(drop=True)  # Concat said reports into a single DataFrame

    def get_raw_timeseries(self):
        return self.transform_jhu_timeseries()

    def get_all_population(self):
        if(self._POPULATION_DATA is None):
            self._POPULATION_DATA = pd.read_csv(self._POPULATION_URL)
        return self._POPULATION_DATA

    def get_all_beds(self):
        if(self._BED_DATA is None):
            self._BED_DATA = pd.read_csv(self._BEDS_URL)
        return self._BED_DATA

    def get_all_states_by_country(self, country):
        return self.get_all_population()[self.get_all_population()[self.COUNTRY_FIELD] == country][self.STATE_FIELD].dropna().unique()


class CDSDataset(Dataset):
    """CoronaDataScraper Dataset"""
    _TIME_SERIES_URL = r'https://coronadatascraper.com/timeseries.csv'
    _POPULATION_URL = r'https://raw.githubusercontent.com/covid-projections/covid-data-model/master/data/populations.csv'
    _BEDS_URL = r'https://raw.githubusercontent.com/covid-projections/covid-data-model/master/data/beds.csv'

    def __init__(self, filter_past_date=None):
        super().__init__(start_date=datetime.datetime(year=2020, month=3, day=3), filter_past_date=filter_past_date)

    def get_raw_timeseries(self):
        if(self._TIME_SERIES_DATA is None):
            self._TIME_SERIES_DATA = pd.read_csv(self._TIME_SERIES_URL, parse_dates=[self.DATE_FIELD])
        return self._TIME_SERIES_DATA

    def get_all_population(self):
        if(self._POPULATION_DATA is None):
            self._POPULATION_DATA = pd.read_csv(self._POPULATION_URL)
        return self._POPULATION_DATA

    def get_all_beds(self):
        if(self._BED_DATA is None):
            self._BED_DATA = pd.read_csv(self._BEDS_URL)
        return self._BED_DATA

    def get_all_states_by_country(self, country):
        return self.get_all_population()[self.get_all_population()[self.COUNTRY_FIELD] == country][self.STATE_FIELD].dropna().unique()
