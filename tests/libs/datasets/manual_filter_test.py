import json
from typing import List
from typing import Optional
from typing import Type

import pytest
from datapublic import common_fields

from datapublic.common_fields import CommonFields
from datapublic.common_fields import DemographicBucket
from datapublic.common_fields import FieldGroup

from cli import data
from libs.datasets import AggregationLevel
from libs.datasets import manual_filter
from libs.datasets import statistical_areas
from libs.datasets import taglib
from libs.datasets import timeseries
from libs.pipeline import Region
from libs.pipeline import RegionMask
from tests import test_helpers
from tests.test_helpers import TimeseriesLiteral


TEST_CONFIG = manual_filter.Config.parse_obj(
    {
        "filters": [
            {
                "regions_included": [
                    Region.from_fips("49009"),
                    Region.from_fips("49013"),
                    Region.from_fips("49047"),
                ],
                "start_date": "2021-02-12",
                "fields_included": [CommonFields.CASES, CommonFields.DEATHS],
                "internal_note": "https://trello.com/c/aj7ep7S7/1130",
                "public_note": "The TriCounty Health Department is focusing on vaccinations "
                "and we have not found a new source of case counts.",
                "drop_observations": True,
            },
            {
                "regions_included": [RegionMask(AggregationLevel.COUNTY, states=["OK"])],
                "regions_excluded": [Region.from_fips("40109"), Region.from_fips("40143")],
                "start_date": "2021-03-15",
                "fields_included": common_fields.FIELD_GROUP_TO_LIST_FIELDS[
                    FieldGroup.CASES_DEATHS
                ],
                "internal_note": "https://trello.com/c/HdAKfp49/1139",
                "public_note": "Something broke with the OK county data.",
                "drop_observations": True,
            },
        ]
    }
)


def test_filter_config_check():
    with pytest.raises(Exception, match="doesn't drop observations or add a public_note"):
        manual_filter.Filter(
            regions_included=[],
            fields_included=[],
            internal_note="",
            public_note="",
            drop_observations=False,
        )

    with pytest.raises(Exception, match="start_date without dropping"):
        manual_filter.Filter(
            regions_included=[],
            fields_included=[],
            internal_note="",
            public_note="We did something",
            start_date="2020-04-01",
            drop_observations=False,
        )


def test_manual_filter():
    r1 = Region.from_fips("49009")
    other_data = {Region.from_fips("06037"): {CommonFields.CASES: [1, 2, 3, 4, 5]}}
    ds_in = test_helpers.build_dataset(
        {r1: {CommonFields.CASES: [4, 5, 6, 7, 8]}, **other_data}, start_date="2021-02-10"
    )

    ds_out = manual_filter.run(ds_in, TEST_CONFIG)

    tag_expected = test_helpers.make_tag(
        taglib.TagType.KNOWN_ISSUE,
        date="2021-02-12",
        public_note=TEST_CONFIG.filters[0].public_note,
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

    ds_out = manual_filter.run(ds_in, TEST_CONFIG)

    tag_expected = test_helpers.make_tag(
        taglib.TagType.KNOWN_ISSUE,
        date="2021-03-15",
        public_note=TEST_CONFIG.filters[1].public_note,
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


def test_manual_filter_field_groups():
    region_included = Region.from_fips("40031")
    other_data = {CommonFields.ICU_BEDS: [3, 4, 5]}
    ds_in = test_helpers.build_default_region_dataset(
        {CommonFields.DEATHS: [1, 2, 3], **other_data},
        start_date="2021-03-14",
        region=region_included,
    )

    ds_out = manual_filter.run(ds_in, TEST_CONFIG)

    tag_expected = test_helpers.make_tag(
        taglib.TagType.KNOWN_ISSUE,
        date="2021-03-15",
        public_note=TEST_CONFIG.filters[1].public_note,
    )
    ds_expected = test_helpers.build_default_region_dataset(
        {CommonFields.DEATHS: TimeseriesLiteral([1], annotation=[tag_expected]), **other_data},
        start_date="2021-03-14",
        region=region_included,
    )

    test_helpers.assert_dataset_like(ds_out, ds_expected)

    ds_out_touched = manual_filter.touched_subset(ds_in, ds_out)
    ds_expected_touched = test_helpers.build_default_region_dataset(
        {CommonFields.DEATHS: TimeseriesLiteral([1, 2, 3], annotation=[tag_expected])},
        start_date="2021-03-14",
        region=region_included,
    )
    test_helpers.assert_dataset_like(ds_out_touched, ds_expected_touched)


def test_manual_filter_per_bucket_tag():
    region = Region.from_fips("40031")
    kids = DemographicBucket("age:0-9")
    guys = DemographicBucket("sex:male")
    all_ = DemographicBucket("all")
    other_data = {
        CommonFields.ICU_BEDS: TimeseriesLiteral([3, 4, 5], annotation=[test_helpers.make_tag()])
    }

    # NOTE(sean): as of 11-29-2021:
    # demographic buckets have data but will not be removed, and should have no tag.
    # non-demographic bucket will have data removed and should have a tag
    ds_in = test_helpers.build_default_region_dataset(
        {CommonFields.CASES: {kids: [1, 2, 3], guys: [4], all_: [5, 6, 7]}, **other_data},
        start_date="2021-03-14",
        region=region,
    )

    ds_out = manual_filter.run(ds_in, TEST_CONFIG)

    tag_expected = test_helpers.make_tag(
        taglib.TagType.KNOWN_ISSUE,
        date="2021-03-15",
        public_note=TEST_CONFIG.filters[1].public_note,
    )
    ds_expected = test_helpers.build_default_region_dataset(
        {
            CommonFields.CASES: {
                kids: [1, 2, 3],
                guys: [4],
                all_: TimeseriesLiteral([5], annotation=[tag_expected]),
            },
            **other_data,
        },
        start_date="2021-03-14",
        region=region,
    )
    test_helpers.assert_dataset_like(ds_out, ds_expected)


def test_region_overrides_transform_smoke_test():
    aggregator = statistical_areas.CountyToCBSAAggregator.from_local_public_data()
    transformed = manual_filter.transform_region_overrides(
        json.load(open(data.REGION_OVERRIDES_JSON)), aggregator.cbsa_to_counties_region_map
    )
    assert transformed.filters
    # pprint.pprint(transformed)


def test_region_overrides_transform_and_filter():
    region_overrides = {
        "overrides": [
            {
                "include": "region-and-subregions",
                "metric": "metrics.vaccinationsInitiatedRatio",
                "region": "WY",
                "context": "https://trello.com/c/kvjwZJJP/1005",
                "disclaimer": "Yo, bad stuff",
                "blocked": True,
            }
        ]
    }

    region = Region.from_state("WY")
    ds_in = test_helpers.build_dataset({region: {CommonFields.VACCINATIONS_INITIATED: [1, 2, 3]}})

    ds_out = manual_filter.run(
        ds_in, manual_filter.transform_region_overrides(region_overrides, {})
    )

    tag_expected = test_helpers.make_tag(
        taglib.TagType.KNOWN_ISSUE_NO_DATE,
        public_note=region_overrides["overrides"][0]["disclaimer"],
    )
    ds_expected = timeseries.MultiRegionDataset.new_without_timeseries().add_tag_to_subset(
        tag_expected, ds_in.timeseries_bucketed_wide_dates.index
    )

    test_helpers.assert_dataset_like(ds_out, ds_expected, drop_na_timeseries=True)


def test_region_overrides_transform_and_filter_blocked_false():
    region_overrides = {
        "overrides": [
            {
                "include": "region-and-subregions",
                "metric": "metrics.caseDensity",
                "region": "TX",
                "context": "https://trello.com/c/kvjwZJJP/1005",
                "disclaimer": "Yo, bad stuff",
                "blocked": False,
            }
        ]
    }

    region_tx = Region.from_state("TX")
    kids = DemographicBucket("age:0-9")
    other_region_metrics = {Region.from_state("AZ"): {CommonFields.CASES: [2, 3]}}
    ds_in = test_helpers.build_dataset(
        {
            region_tx: {
                CommonFields.CASES: {DemographicBucket.ALL: [6, 8], kids: [1, 2]},
                CommonFields.ICU_BEDS: [5, 5],
            },
            **other_region_metrics,
        }
    )

    ds_out = manual_filter.run(
        ds_in, manual_filter.transform_region_overrides(region_overrides, {})
    )

    tag = test_helpers.make_tag(
        taglib.TagType.KNOWN_ISSUE_NO_DATE,
        public_note=region_overrides["overrides"][0]["disclaimer"],
    )
    # NOTE(sean): as of 11-29-2021:
    # demographic buckets are not blocked, and therefore should not have an annotation tag
    ds_expected = test_helpers.build_dataset(
        {
            region_tx: {
                CommonFields.CASES: {
                    DemographicBucket.ALL: TimeseriesLiteral([6, 8], annotation=[tag]),
                    kids: [1, 2],
                },
                CommonFields.ICU_BEDS: [5, 5],
            },
            **other_region_metrics,
        }
    )

    test_helpers.assert_dataset_like(ds_out, ds_expected)


def test_region_overrides_transform_and_filter_infection_rate():
    """Test that metrics.infectionRate doesn't cause a crash."""
    region_overrides = {
        "overrides": [
            {
                "include": "region",
                "metric": "metrics.infectionRate",
                "region": "TX",
                "context": "https://foo.com/",
                "disclaimer": "Blah",
                "blocked": True,
            }
        ]
    }

    ds_in = test_helpers.build_default_region_dataset(
        {CommonFields.CASES: [6, 8]}, region=Region.from_state("TX")
    )

    ds_out = manual_filter.run(
        ds_in, manual_filter.transform_region_overrides(region_overrides, {})
    )

    test_helpers.assert_dataset_like(ds_out, ds_in)


@pytest.mark.parametrize(
    "start_date, end_date, result, tag_date, tag_second_date",
    [
        ("2020-04-03", None, [1, 2, None, None], "2020-04-03", None),
        (None, "2020-04-02", [None, None, 3, 4], "2020-04-02", None),
        ("2020-04-02", "2020-04-03", [1, None, None, 4], "2020-04-02", "2020-04-03"),
        ("2020-04-02", "2020-04-02", [1, None, 3, 4], "2020-04-02", "2020-04-02"),
        ("2020-04-05", None, [1, 2, 3, 4], None, None),
        (None, "2020-03-28", [1, 2, 3, 4], None, None),
        (None, "2020-04-05", [None, None, None, None], "2020-04-05", None),
        ("2020-03-28", None, [None, None, None, None], "2020-03-28", None),
    ],
)
def test_region_overrides_transform_and_filter_start_end_dates(
    start_date: Optional[str],
    end_date: Optional[str],
    result: List,
    tag_date: Optional[str],
    tag_second_date: Optional[str],
):
    region_overrides = {
        "overrides": [
            {
                "include": "region",
                "metric": "metrics.caseDensity",
                "region": "TX",
                "context": "https://foo.com/",
                "disclaimer": "Blah",
                "blocked": True,
                "start_date": start_date,
                "end_date": end_date,
            }
        ]
    }
    region_tx = Region.from_state("TX")

    ds_in = test_helpers.build_default_region_dataset(
        {CommonFields.CASES: [1, 2, 3, 4]}, region=region_tx
    )

    ds_out = manual_filter.run(
        ds_in, manual_filter.transform_region_overrides(region_overrides, {})
    )

    if tag_date and tag_second_date:
        tags = [
            test_helpers.make_tag(
                taglib.TagType.KNOWN_ISSUE_DATE_RANGE,
                public_note="Blah",
                start_date=tag_date,
                end_date=tag_second_date,
            )
        ]
    elif tag_date:
        tags = [
            test_helpers.make_tag(taglib.TagType.KNOWN_ISSUE, public_note="Blah", date=tag_date)
        ]
    else:
        tags = []
    ds_expected = test_helpers.build_default_region_dataset(
        {CommonFields.CASES: TimeseriesLiteral(result, annotation=tags)}, region=region_tx,
    )
    test_helpers.assert_dataset_like(ds_out, ds_expected, drop_na_timeseries=True)


def test_region_overrides_transform_and_filter_override_with_multiple_regions():
    region_overrides = {
        "overrides": [
            {
                "include": "region-and-subregions",
                "metric": "metrics.caseDensity",
                "region": "WY, TX",
                "context": "",
                "disclaimer": "Bad data",
                "blocked": False,
            }
        ]
    }

    region_wy = Region.from_state("WY")
    region_tx = Region.from_state("TX")
    region_ca = Region.from_state("CA")
    ds_in = test_helpers.build_dataset(
        {
            region_wy: {CommonFields.CASES: [1, 2, 3]},
            region_tx: {CommonFields.CASES: [4, 5, 6]},
            region_ca: {CommonFields.CASES: [7, 8, 9]},
        }
    )

    ds_out = manual_filter.run(
        ds_in, manual_filter.transform_region_overrides(region_overrides, {})
    )

    tag_expected = test_helpers.make_tag(
        taglib.TagType.KNOWN_ISSUE_NO_DATE,
        public_note=region_overrides["overrides"][0]["disclaimer"],
    )
    ds_expected = test_helpers.build_dataset(
        {
            region_wy: {
                CommonFields.CASES: TimeseriesLiteral([1, 2, 3], annotation=[tag_expected])
            },
            region_tx: {
                CommonFields.CASES: TimeseriesLiteral([4, 5, 6], annotation=[tag_expected])
            },
            region_ca: {CommonFields.CASES: [7, 8, 9]},
        }
    )

    test_helpers.assert_dataset_like(ds_out, ds_expected, drop_na_timeseries=True)


@pytest.mark.parametrize(
    "blocked, disclaimer, start_date, end_date, raises",
    [
        # Not blocked and no disclaimer is invalid.
        (False, None, None, None, ValueError),
        # Setting disclaimer fixes the problem so transform_region_overrides returns normally.
        (False, "Stuff", None, None, None),
        # start_date or end_date with blocked: False is a problem
        (False, "Stuff", "2020-04-01", None, ValueError),
        (False, "Stuff", None, "2020-04-01", ValueError),
        # start_date or end_date with blocked: True works
        (True, "Stuff", "2020-04-01", None, None),
        (True, "Stuff", None, "2020-04-01", None),
        # Good start_date and end_date
        (True, "Stuff", "2020-04-01", "2020-04-02", None),
        # Bad start_date and end_date
        (True, "Stuff", "2020-04-02", "2020-04-01", ValueError),
    ],
)
def test_region_overrides_transform_and_filter_validation(
    blocked: bool,
    disclaimer: Optional[str],
    start_date: Optional[str],
    end_date: Optional[str],
    raises: Optional[Type[BaseException]],
):
    region_overrides = {
        "overrides": [
            {
                "include": "region",
                "metric": "metrics.caseDensity",
                "region": "TX",
                "context": "https://foo.com/",
                "blocked": blocked,
            }
        ]
    }
    if disclaimer:
        region_overrides["overrides"][0]["disclaimer"] = disclaimer
    if start_date:
        region_overrides["overrides"][0]["start_date"] = start_date
    if end_date:
        region_overrides["overrides"][0]["end_date"] = end_date

    if raises:
        with pytest.raises(raises):
            manual_filter.transform_region_overrides(region_overrides, {})
    else:
        manual_filter.transform_region_overrides(region_overrides, {})


def test_block_removes_existing_source_tag():
    source = taglib.Source("TestSource")
    other_metrics = {CommonFields.DEATHS: [0, 0]}
    ds_in = test_helpers.build_default_region_dataset(
        {CommonFields.CASES: TimeseriesLiteral([1, 2], source=source), **other_metrics}
    )

    config = manual_filter.Config(
        filters=[
            manual_filter.Filter(
                regions_included=[test_helpers.DEFAULT_REGION],
                fields_included=[CommonFields.CASES],
                drop_observations=True,
                internal_note="",
                public_note="We fixed it",
            )
        ]
    )
    ds_out = manual_filter.run(ds_in, config)

    tag_expected = test_helpers.make_tag(
        taglib.TagType.KNOWN_ISSUE_NO_DATE, public_note=config.filters[0].public_note,
    )
    ds_expected = test_helpers.build_default_region_dataset(
        {
            CommonFields.CASES: TimeseriesLiteral([None, None], annotation=[tag_expected]),
            **other_metrics,
        }
    )

    test_helpers.assert_dataset_like(ds_out, ds_expected, drop_na_timeseries=True)

    ds_touched = manual_filter.touched_subset(ds_in, ds_out)
    ds_touched_expected = test_helpers.build_default_region_dataset(
        {CommonFields.CASES: TimeseriesLiteral([1, 2], source=source, annotation=[tag_expected])}
    )
    test_helpers.assert_dataset_like(ds_touched, ds_touched_expected)


def test_touched_subset():
    source = taglib.Source("TestSource")
    other_metrics = {CommonFields.DEATHS: [0, 0]}
    ds_in = test_helpers.build_default_region_dataset(
        {CommonFields.CASES: TimeseriesLiteral([1, 2], source=source), **other_metrics}
    )
    # Add 2 known issue tags for the same time series and check that touched_subset still produces
    # the expected output.
    known_issue_1 = test_helpers.make_tag(
        taglib.TagType.KNOWN_ISSUE, public_note="foo", date="2021-04-01"
    )
    known_issue_2 = test_helpers.make_tag(
        taglib.TagType.KNOWN_ISSUE, public_note="foo", date="2021-04-02"
    )
    ds_out = test_helpers.build_default_region_dataset(
        {
            CommonFields.CASES: TimeseriesLiteral(
                [1, None], source=source, annotation=[known_issue_1, known_issue_2]
            ),
            **other_metrics,
        }
    )
    ds_touched = manual_filter.touched_subset(ds_in, ds_out)
    ds_touched_expected = test_helpers.build_default_region_dataset(
        {
            CommonFields.CASES: TimeseriesLiteral(
                [1, 2], source=source, annotation=[known_issue_1, known_issue_2]
            )
        }
    )
    test_helpers.assert_dataset_like(ds_touched, ds_touched_expected)


def test_touched_subset_only_observation_drops():
    # (Ab)use the demographic buckets as a way to identify time series.
    bucket_no_issue = DemographicBucket("b1")
    bucket_has_tag = DemographicBucket("b2")
    # Make 6 time series. half will get a KNOWN_ISSUE tag, half won't.
    ds_in = test_helpers.build_default_region_dataset(
        {
            CommonFields.CASES: {bucket_no_issue: [1, 2], bucket_has_tag: [3, 4]},
            CommonFields.DEATHS: {bucket_no_issue: [5, 6], bucket_has_tag: [7, 8]},
            CommonFields.ICU_BEDS: {bucket_no_issue: [9, 1], bucket_has_tag: [2, 3]},
        }
    )
    known_issue = test_helpers.make_tag(
        taglib.TagType.KNOWN_ISSUE, public_note="foo", date="2021-04-01"
    )
    # ds_out is a mock of what is returned by `manual_filter.run`.
    ds_out = test_helpers.build_default_region_dataset(
        {
            # Drop all observations of CASES.
            CommonFields.CASES: {
                bucket_no_issue: [None, None],
                bucket_has_tag: TimeseriesLiteral([None, None], annotation=[known_issue]),
            },
            # Drop one observation of each DEATHS time series.
            CommonFields.DEATHS: {
                bucket_no_issue: [5, None],
                bucket_has_tag: TimeseriesLiteral([7, None], annotation=[known_issue]),
            },
            # Drop no observations of ICU_BEDS.
            CommonFields.ICU_BEDS: {
                bucket_no_issue: [9, 1],
                bucket_has_tag: TimeseriesLiteral([2, 3], annotation=[known_issue]),
            },
        }
    )
    ds_touched = manual_filter.touched_subset(ds_in, ds_out)
    ds_touched_expected = test_helpers.build_default_region_dataset(
        {
            # Only time series that had the tag and at least one dropped observation are put in
            # the touched dataset.
            CommonFields.CASES: {
                bucket_has_tag: TimeseriesLiteral([3, 4], annotation=[known_issue]),
            },
            CommonFields.DEATHS: {
                bucket_has_tag: TimeseriesLiteral([7, 8], annotation=[known_issue]),
            },
        }
    )
    test_helpers.assert_dataset_like(ds_touched, ds_touched_expected)
