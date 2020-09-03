import pytest

from libs import pipeline
from pyseir.icu import infer_icu


# Check some counties picked arbitrarily: San Francisco/06075, Houston (Harris County, TX)/48201, Riley KS/20161
@pytest.mark.parametrize("fips", ["06075", "48201", "20161"])
def test_get_icu(fips):
    region = pipeline.Region.from_fips(fips)
    regional_combined_data = pipeline.RegionalCombinedData.from_region(region)
    state_combined_data = pipeline.RegionalCombinedData.from_region(region.get_state_region())
    infer_icu.get_icu_timeseries(
        region,
        regional_combined_data=regional_combined_data.get_timeseries(),
        state_combined_data=state_combined_data.get_timeseries(),
    )
