from covidactnow.datapublic.common_fields import CommonFields
from covidactnow.datapublic.common_fields import DemographicBucket

from libs.datasets import manual_filter
from libs.datasets import taglib
from libs.pipeline import Region
from tests import test_helpers
from tests.test_helpers import TimeseriesLiteral


def test_manual_filter():
    r1 = Region.from_fips("49009")
    other_data = {Region.from_fips("06037"): {CommonFields.CASES: [1, 2, 3, 4, 5]}}
    ds_in = test_helpers.build_dataset(
        {r1: {CommonFields.CASES: [4, 5, 6, 7, 8]}, **other_data}, start_date="2021-02-10"
    )

    ds_out = manual_filter.run(ds_in, manual_filter.CONFIG)

    tag_expected = test_helpers.make_tag(
        taglib.TagType.KNOWN_ISSUE,
        date="2021-02-12",
        disclaimer=manual_filter.CONFIG["filters"][0]["drop_observations"][0]["public_note"],
    )
    ds_expected = test_helpers.build_dataset(
        {
            r1: {CommonFields.CASES: TimeseriesLiteral([4, 5], annotation=[tag_expected])},
            **other_data,
        },
        start_date="2021-02-10",
    )

    test_helpers.assert_dataset_like(ds_out, ds_expected)


def test_manual_filter_region_excluded():
    region_included = Region.from_fips("40031")
    region_excluded = Region.from_fips("40109")
    other_data = {region_excluded: {CommonFields.CASES: [3, 4, 5]}}
    ds_in = test_helpers.build_dataset(
        {region_included: {CommonFields.CASES: [1, 2, 3]}, **other_data}, start_date="2021-03-14"
    )

    ds_out = manual_filter.run(ds_in, manual_filter.CONFIG)

    tag_expected = test_helpers.make_tag(
        taglib.TagType.KNOWN_ISSUE,
        date="2021-03-15",
        disclaimer=manual_filter.CONFIG["filters"][1]["drop_observations"][0]["public_note"],
    )
    ds_expected = test_helpers.build_dataset(
        {
            region_included: {
                CommonFields.CASES: TimeseriesLiteral([1], annotation=[tag_expected])
            },
            **other_data,
        },
        start_date="2021-03-14",
    )

    test_helpers.assert_dataset_like(ds_out, ds_expected)


def test_manual_filter_per_bucket_tag():
    region = Region.from_fips("40031")
    kids = DemographicBucket("age:0-9")
    guys = DemographicBucket("sex:male")
    other_data = {
        CommonFields.ICU_BEDS: TimeseriesLiteral([3, 4, 5], annotation=[test_helpers.make_tag()])
    }
    # kids have real values that will be removed so will have a tag. guys do not and are not
    # expected to get a tag.
    ds_in = test_helpers.build_default_region_dataset(
        {CommonFields.CASES: {kids: [1, 2, 3], guys: [4]}, **other_data},
        start_date="2021-03-14",
        region=region,
    )

    ds_out = manual_filter.run(ds_in, manual_filter.CONFIG)

    tag_expected = test_helpers.make_tag(
        taglib.TagType.KNOWN_ISSUE,
        date="2021-03-15",
        disclaimer=manual_filter.CONFIG["filters"][1]["drop_observations"][0]["public_note"],
    )
    ds_expected = test_helpers.build_default_region_dataset(
        {
            CommonFields.CASES: {
                kids: TimeseriesLiteral([1], annotation=[tag_expected]),
                guys: [4],
            },
            **other_data,
        },
        start_date="2021-03-14",
        region=region,
    )

    test_helpers.assert_dataset_like(ds_out, ds_expected)
