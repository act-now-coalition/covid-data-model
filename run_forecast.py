import pandas as pd
from pyseir.inference import forecast_rt
from covidactnow.datapublic.common_fields import CommonFields

# from libs.datasets import combined_datasets
# from libs.datasets.timeseries import TimeseriesDataset
# from libs.datasets.dataset_utils import AggregationLevel


# print("starting")
# all_data = combined_datasets.build_us_timeseries_with_all_fields().get_data(
#    AggregationLevel.STATE, country="USA"
# )
# print(all_data)
# all_data.to_csv("all_combined_datasets.csv")
# print("got data")

forecast_rt.external_run_forecast()
print("done")
