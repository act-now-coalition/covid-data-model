import pandas as pd
from covid_data_model import run


def test_run_simulation():
    beds = pd.read_csv("covid_data_model/data/beds.csv")
    populations = pd.read_csv("covid_data_model/data/populations.csv")
    full_timeseries = pd.read_csv("covid_data_model/data/timeseries.csv")
    country = "USA"
    run.main(beds, populations, full_timeseries, country)
