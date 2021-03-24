from covidactnow.datapublic.common_fields import CommonFields

from libs.datasets import manual_filter
from libs.pipeline import Region
from tests import test_helpers


def test_manual_filter():
    r1 = Region.from_fips("49009")
    other_data = {Region.from_fips("06037"): {CommonFields.CASES: list(range(10, 20))}}
    ds_in = test_helpers.build_dataset(
        {r1: {CommonFields.CASES: list(range(10))}, **other_data}, start_date="2021-02-10"
    )

    ds_out = manual_filter.run(ds_in, manual_filter.CONFIG)

    ds_expected = test_helpers.build_dataset(
        {r1: {CommonFields.CASES: list(range(2))}, **other_data}, start_date="2021-02-10"
    )

    test_helpers.assert_dataset_like(ds_out, ds_expected)
