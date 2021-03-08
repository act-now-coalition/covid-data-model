from covidactnow.datapublic.common_fields import CommonFields

from libs import pipeline
from libs.datasets.sources import cdc_testing_dataset
from tests import test_helpers


def test_remove_trailing_zeros():
    region_dc_county = pipeline.Region.from_fips("11001")
    region_dc_state = pipeline.Region.from_state("DC")
    region_sf = pipeline.Region.from_fips("06075")

    # Other data, in the test to make sure it isn't dropped
    other_regions_data = {
        region_sf: {CommonFields.TEST_POSITIVITY_7D: [10, 20, 30, 40, 50, 60]},
    }
    dc_data_before_filter = {
        CommonFields.TEST_POSITIVITY_7D: [50, 60, 0, 0, 0, 0],
        CommonFields.CASES: [1, 2, 3, 4, 5, 6],
    }
    ds_in = test_helpers.build_dataset(
        {region_dc_county: dc_data_before_filter, **other_regions_data,}
    )

    ds_out = cdc_testing_dataset.modify_dataset(ds_in)

    dc_data_after_filter = {
        CommonFields.TEST_POSITIVITY_7D: [0.5, 0.6, None, None, None, None],
        CommonFields.CASES: [1, 2, 3, 4, 5, 6],
    }
    ds_expected = test_helpers.build_dataset(
        {
            region_dc_county: dc_data_after_filter,
            region_dc_state: dc_data_after_filter,
            **other_regions_data,
        }
    )

    test_helpers.assert_dataset_like(ds_out, ds_expected, drop_na_dates=True)
