import dataclasses

from covidactnow.datapublic.common_fields import CommonFields

from libs.datasets.sources import cdc_testing_dataset
from tests import test_helpers


def test_remove_trailing_zeros():

    ds_in = test_helpers.build_default_region_dataset(
        {CommonFields.TEST_POSITIVITY_7D: [0.5, 0.6, 0, 0, 0, 0]}
    )

    ds_out = dataclasses.replace(
        ds_in, timeseries=cdc_testing_dataset.remove_trailing_zeros(ds_in.timeseries)
    )

    ds_expected = test_helpers.build_default_region_dataset(
        {CommonFields.TEST_POSITIVITY_7D: [0.5, 0.6]}
    )

    test_helpers.assert_dataset_like(ds_out, ds_expected, drop_na_dates=True)
