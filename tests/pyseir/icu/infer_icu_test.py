import pytest

import tests.test_helpers
from covidactnow.datapublic.common_fields import CommonFields

from libs import pipeline
from pyseir.icu import infer_icu
from pyseir.icu.infer_icu import ICUWeightsPath


@pytest.mark.parametrize(
    "fips",
    [
        "06075",  # San Francisco
        "48201",  # Houston (Harris County, TX)
        "20161",  # Riley KS (small, takes a different code path).
    ],
)
def test_get_icu(fips):
    test_dataset = tests.test_helpers.load_test_dataset()
    region = pipeline.Region.from_fips(fips)
    regional_combined_data_timeseries = test_dataset.get_one_region(region)
    state_combined_data_timeseries = test_dataset.get_one_region(region.get_state_region())
    # TODO(tom): Replace with call to get_icu_timeseries_from_regional_input which is what is
    #  used it non-test code, then delete get_icu_timeseries
    result_series = infer_icu.get_icu_timeseries(
        region,
        regional_combined_data=regional_combined_data_timeseries,
        state_combined_data=state_combined_data_timeseries,
        weight_by=ICUWeightsPath.ONE_MONTH_TRAILING_CASES,
    )
    assert not result_series.dropna().empty
    assert result_series.index.names == [CommonFields.DATE]
