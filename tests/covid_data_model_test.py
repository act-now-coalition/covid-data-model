import pandas as pd
from covid_data_model import run


def test_run_simulation():
    beds = pd.read_csv("data/beds.csv")
    populations = pd.read_csv("data/populations.csv")
    full_timeseries = pd.read_csv("data/timeseries.csv")
    country = "USA"
    run.main(beds, populations, full_timeseries, country)
