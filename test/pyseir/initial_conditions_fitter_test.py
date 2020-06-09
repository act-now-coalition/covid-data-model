import pandas as pd
import numpy
from pyseir.inference import initial_conditions_fitter
from libs.datasets import combined_datasets


def test_fips_metadata(nyc_fips):
    combined_datasets.build_us_timeseries_with_all_fields()
    fitter = initial_conditions_fitter.InitialConditionsFitter(nyc_fips)
    assert fitter.state == "NY"
    assert fitter.county == "New York County"
    assert fitter.data_start_date == pd.Timestamp("2020-03-01")
    # Checking to make sure that y is a numpy array rather than a pandas DF.
    assert isinstance(fitter.y, numpy.ndarray)
