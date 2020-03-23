import datetime
import logging
import math
from copy import copy

import pandas as pd
class CovidDatasets:

	## Constants
	POPULATION_URL = 'https://raw.githubusercontent.com/covid-projections/covid-data-model/master/data/populations.csv'
	BED_URL = 'https://raw.githubusercontent.com/covid-projections/covid-data-model/master/data/beds.csv'
	TIME_SERIES_URL = "https://coronadatascraper.com/timeseries.csv"
	DATE_FIELD = "date"
	TIME_SERIES_DATA = None
	BED_DATA = None
	POPULATION_DATA = None
	start_date = datetime.datetime(year=2020, month=3, day=3)
	# Initializer / Instance Attributes
	def __init__(self):
		logging.basicConfig(level=logging.CRITICAL)

	def backfill_to_init_date(self, series):
		# We need to make sure that the data starts from Mar3, no matter when our records begin
		init_data_date = series[self.DATE_FIELD].min()
		synthetic_interval = (init_data_date - self.start_date).days # The number of days we need to create to backfill to Mar3
		template = series.iloc[0] # grab a row for us to copy structure and data from
		synthetic_data = []
		pd.set_option('mode.chained_assignment', None) # Suppress pandas' anxiety. We know what we're doing
		for i in range(0, synthetic_interval):
			synthetic_row = template
			synthetic_row[self.DATE_FIELD] = self.start_date + datetime.timedelta(days=i)
			synthetic_row['cases'] = 0
			synthetic_row['deaths'] = 0
			synthetic_row['recovered'] = 0
			synthetic_row['active'] = 0
			synthetic_data.append(copy(synthetic_row)) # We need to copy it to prevent alteration by reference
		pd.set_option('mode.chained_assignment', 'warn') # Turn the anxiety back on
		synthetic_series = pd.DataFrame(synthetic_data)
		new_series = series.append(synthetic_series)
		return new_series.sort_values('date').reset_index()

	def backfill_synthetic_cases(self, series):
		# Fill in all values prior to the first non-zero values. Use 1/2 following value. Decays into nothing
		# sort the dataframe in reverse date order, so we traverse from latest to earliest
		series = series.sort_values(self.DATE_FIELD, ascending=False).reset_index()
		for i in range(0, len(series)):
			if series.iloc[i]['cases'] == 0:
				series.at[i, 'cases'] = math.floor(series.iloc[i - 1]['cases'] / 2)
		return series

	def backfill(self, series):
		# Backfill the data as necessary for the model
		return self.backfill_synthetic_cases(
			self.backfill_to_init_date(series)
		)

	def cutoff(self, series):
		# The model MUST start on a certain day. If there is data that precedes that date,
		#  we must trim it from the series
		return series[series[self.DATE_FIELD] >= self.start_date]

	def prep_data(self, series):
		# We have some requirements of the data's format, and that is enforced here.
		return self.cutoff(self.backfill(series))

	def get_all_timeseries(self):
		if(self.TIME_SERIES_DATA is None):
			self.TIME_SERIES_DATA = pd.read_csv(self.TIME_SERIES_URL)
			self.TIME_SERIES_DATA[self.DATE_FIELD] = pd.to_datetime(self.TIME_SERIES_DATA[self.DATE_FIELD])
		return self.TIME_SERIES_DATA

	def get_all_population(self):
		if(self.POPULATION_DATA is None):
			self.POPULATION_DATA = pd.read_csv(self.POPULATION_URL)
		return self.POPULATION_DATA

	def get_all_beds(self):
		if(self.BED_DATA is None):
			self.BED_DATA = pd.read_csv(self.BED_URL)
		return self.BED_DATA

	def get_timeseries_by_country_state(self, country, state):
		# First, attempt to pull the state-level data without aggregating.
		return self.backfill(
			self.get_all_timeseries()[
				(self.get_all_timeseries()["state"] == state) &
				(self.get_all_timeseries()["country"] == country) &
				(self.get_all_timeseries()["county"].isna())
			]
		)

	def get_timeseries_by_country(self, country):
		return self.get_all_timeseries()[self.get_all_timeseries()["country"] == country]

	def get_population_by_country_state(self, country, state):
		matching_pops = self.get_all_population()[(self.get_all_population()["state"] == state) & (
		self.get_all_population()["country"] == country)]
		return int(matching_pops.iloc[0].at["population"])

	def get_beds_by_country_state(self, country, state):
		matching_beds = self.get_all_beds()[(self.get_all_beds()["state"] == state) &
								  (self.get_all_beds()["country"] == country)]
		beds_per_mille = float(matching_beds.iloc[0].at["bedspermille"])
		return int(round(beds_per_mille * self.get_population_by_country_state(country, state) / 1000))
