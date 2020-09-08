import pytest
from covidactnow.datapublic.common_fields import CommonFields

from libs import pipeline
from pyseir.icu import infer_icu


@pytest.mark.parametrize(
    "fips",
    [
        "06075",  # San Francisco
        "48201",  # Houston (Harris County, TX)
        "20161",  # Riley KS (small, takes a different code path).
    ],
)
def test_get_icu(fips):
    region = pipeline.Region.from_fips(fips)
    regional_combined_data = pipeline.RegionalCombinedData.from_region(region)
    state_combined_data = pipeline.RegionalCombinedData.from_region(region.get_state_region())
    result_series = infer_icu.get_icu_timeseries(
        region,
        regional_combined_data=regional_combined_data.get_timeseries(),
        state_combined_data=state_combined_data.get_timeseries(),
    )
    assert not result_series.dropna().empty
    assert result_series.index.names == [CommonFields.DATE]
