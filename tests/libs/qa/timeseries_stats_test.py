from datapublic.common_fields import CommonFields
from datapublic.common_fields import DemographicBucket
from datapublic.common_fields import FieldGroup
from datapublic.common_fields import PdFields

from libs.datasets import AggregationLevel
from libs.datasets import taglib
from libs.datasets import timeseries
from libs.datasets.taglib import UrlStr
from libs.pipeline import Region
from libs.qa import timeseries_stats
from libs.qa.timeseries_stats import StatName
from tests import test_helpers
from tests.test_helpers import TimeseriesLiteral
import pandas as pd
import numpy as np


def test_make_from_dataset():
    region_tx = Region.from_state("TX")
    region_sf = Region.from_fips("06075")
    region_la = Region.from_fips("06037")
    bucket_40s = DemographicBucket("age:40-49")
    tag1 = test_helpers.make_tag(date="2020-04-01")
    tag2a = test_helpers.make_tag(date="2020-04-02")
    tag2b = test_helpers.make_tag(date="2020-04-03")
    source = taglib.Source("NYTimes", url=UrlStr("http://datasource.co/"))

    dataset = test_helpers.build_dataset(
        {
            region_tx: {
                CommonFields.CASES: [1, 2, 3],
                CommonFields.VACCINATIONS_COMPLETED: {
                    bucket_40s: TimeseriesLiteral([6, 7], annotation=[tag1]),
                    DemographicBucket.ALL: [7, 8, 9],
                },
            },
            # `PerVariable.make` crashes if there are no Source tags anywhere.
            region_sf: {
                CommonFields.CASES: (
                    TimeseriesLiteral([4, 5], annotation=[tag2a, tag2b], source=source)
                )
            },
            region_la: {CommonFields.CASES: [8, 9]},
        },
        static_by_region_then_field_name={
            region_tx: {CommonFields.POPULATION: 10_000},
            region_sf: {CommonFields.POPULATION: 1_000},
            region_la: {CommonFields.POPULATION: 5_000},
        },
    )
    dataset = timeseries.make_source_url_tags(dataset)

    per_timeseries = timeseries_stats.PerTimeseries.make(dataset)

    per_region = per_timeseries.aggregate(CommonFields.LOCATION_ID)
    assert per_region.stats.at[region_tx.location_id, StatName.HAS_TIMESERIES] == 3
    assert per_region.stats.at[region_sf.location_id, StatName.ANNOTATION_COUNT] == 2
    assert per_region.stats.at[region_tx.location_id, StatName.ANNOTATION_COUNT] == 1
    assert per_region.stats.at[region_tx.location_id, StatName.OBSERVATION_COUNT] == 8
    assert per_region.stats.at[region_la.location_id, StatName.OBSERVATION_COUNT] == 2

    cases_by_level = per_timeseries.subset_variables([CommonFields.CASES]).aggregate(
        CommonFields.AGGREGATE_LEVEL
    )
    assert cases_by_level.stats.at[AggregationLevel.COUNTY.value, StatName.HAS_TIMESERIES] == 2
    assert cases_by_level.stats.at[AggregationLevel.COUNTY.value, StatName.SOURCE] == 1
    assert cases_by_level.stats.at[AggregationLevel.COUNTY.value, StatName.ANNOTATION_COUNT] == 2

    by_fieldgroup = per_timeseries.aggregate(timeseries_stats.FIELD_GROUP)
    assert by_fieldgroup.stats.at[FieldGroup.CASES_DEATHS, StatName.HAS_TIMESERIES] == 3
    assert by_fieldgroup.stats.at[FieldGroup.CASES_DEATHS, StatName.SOURCE] == 1
    assert by_fieldgroup.stats.at[FieldGroup.CASES_DEATHS, StatName.ANNOTATION_COUNT] == 2
    assert by_fieldgroup.stats.at[FieldGroup.VACCINES, StatName.OBSERVATION_COUNT] == 5

    assert not per_timeseries.stats_for_locations(dataset.location_ids).empty
    assert per_timeseries.pivottable_data


def test_enum_names_match_values():
    test_helpers.assert_enum_names_match_values(timeseries_stats.StatName)


def test_tag_type_counts():
    bucket_40s = DemographicBucket("age:40-49")
    tag1 = test_helpers.make_tag(taglib.TagType.CUMULATIVE_TAIL_TRUNCATED, date="2020-04-01")
    tag2 = test_helpers.make_tag(taglib.TagType.ZSCORE_OUTLIER)
    source = taglib.Source("NYTimes", url=UrlStr("http://datasource.co/"))

    dataset = test_helpers.build_default_region_dataset(
        {
            CommonFields.CASES: {
                DemographicBucket.ALL: TimeseriesLiteral([1], source=source),
                bucket_40s: TimeseriesLiteral([1], annotation=[tag1, tag2]),
            },
            CommonFields.ICU_BEDS: TimeseriesLiteral([1], annotation=[tag1, tag1]),
        }
    )
    dataset = timeseries.make_source_url_tags(dataset)

    stats = timeseries_stats.PerTimeseries.make(dataset)

    tag_stats_expected = test_helpers.flatten_3_nested_dict(
        {
            CommonFields.CASES: {
                DemographicBucket.ALL: {taglib.TagType.SOURCE: 1, taglib.TagType.SOURCE_URL: 1},
                bucket_40s: {tag1.tag_type: 1, tag2.tag_type: 1},
            },
            CommonFields.ICU_BEDS: {DemographicBucket.ALL: {tag1.tag_type: 2}},
        },
        index_names=[PdFields.VARIABLE, PdFields.DEMOGRAPHIC_BUCKET, "stat_name"],
    ).sort_index()

    tag_stats = (
        stats.stats.loc[:, list(taglib.TagType)]
        .rename_axis(columns="stat_name")
        .stack()
        .replace({0: np.nan})
        .dropna()
    ).sort_index()
    tag_stats = tag_stats.droplevel(
        tag_stats.index.names.difference(tag_stats_expected.index.names)
    )

    pd.testing.assert_series_equal(tag_stats, tag_stats_expected, check_dtype=False)
