import logging
import pandas as pd
class CovidDatasets:

	## Constants
	POPULATION_URL = 'https://raw.githubusercontent.com/covid-projections/covid-data-model/master/data/populations.csv'
	BED_URL = 'https://raw.githubusercontent.com/covid-projections/covid-data-model/master/data/beds.csv'
	TIME_SERIES_URL = "https://coronadatascraper.com/timeseries.csv"
	TIME_SERIES_DATA = None
	BED_DATA = None
	POPULATION_DATA = None
	# Initializer / Instance Attributes
	def __init__(self):
		logging.basicConfig(level=logging.CRITICAL)


	def get_all_timeseries(self):
		if(self.TIME_SERIES_DATA is None):
			self.TIME_SERIES_DATA = pd.read_csv(self.TIME_SERIES_URL)
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
		filtered_timeseries = self.get_all_timeseries()[(self.get_all_timeseries()["state"] == state) & (
		    self.get_all_timeseries()["country"] == country) & (self.get_all_timeseries()["county"].isna())]

	def get_timeseries_by_country(self, country):
	      filtered_timeseries = self.get_all_timeseries()[(self.get_all_timeseries()["state"] == state) & (
	          self.get_all_timeseries()["country"] == country)]

	def get_population_by_country_state(self, country, state):
			matching_pops = self.get_all_population()[(self.get_all_population()["state"] == state) & (
		    self.get_all_population()["country"] == country)]
			return int(matching_pops.iloc[0].at["population"])


	def get_beds_by_country_state(self, country, state):
		matching_beds = self.get_all_beds()[(self.get_all_beds()["state"] == state) &
		                          (self.get_all_beds()["country"] == country)]
		beds_per_mille = float(matching_beds.iloc[0].at["bedspermille"])
		return int(round(beds_per_mille * self.get_population_by_country_state(country, state) / 1000))
