import logging
import pandas as pd
class CovidDatasets:

	## Constants
	POPULATION_URL = 'https://raw.githubusercontent.com/covid-projections/covid-data-model/master/data/populations.csv'
	BED_URL = 'https://raw.githubusercontent.com/covid-projections/covid-data-model/master/data/beds.csv'
	TIME_SERIES_URL = "https://coronadatascraper.com/timeseries.csv"

	# Initializer / Instance Attributes
	def __init__(self):
		logging.basicConfig(level=logging.CRITICAL)


	def get_timeseries(self):
		return pd.read_csv(self.TIME_SERIES_URL)

	def get_population(self):
		return pd.read_csv(self.POPULATION_URL)

	def get_beds(self):
		return pd.read_csv(self.BED_URL)
