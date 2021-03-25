from covidactnow.datapublic.common_fields import CommonFields
from covidactnow.datapublic.common_fields import DemographicBucket
from covidactnow.datapublic.common_fields import FieldGroup

from dash_app.dashboard import population_ratio_by_variable
from libs.datasets import AggregationLevel
from libs.datasets import taglib
from libs.datasets import timeseries
from libs.datasets.taglib import UrlStr
from libs.pipeline import Region
from libs.qa import timeseries_stats
from tests import test_helpers
from tests.test_helpers import TimeseriesLiteral


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
            # `PerTimeseriesStats.make` crashes if there are no Source tags anywhere.
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

    per_region = timeseries_stats.PerTimeseriesStats.make(dataset).aggregate_buckets()

    assert (
        per_region.has_timeseries.at[region_tx.location_id, CommonFields.VACCINATIONS_COMPLETED]
        == 2
    )
    assert per_region.annotation_count.at[region_sf.location_id, CommonFields.CASES] == 2

    cases_by_level = per_region.subset_variables([CommonFields.CASES]).aggregate(
        timeseries_stats.RegionAggregationMethod.LEVEL,
        timeseries_stats.VariableAggregationMethod.NONE,
    )
    assert cases_by_level.has_timeseries.at[AggregationLevel.COUNTY.value, CommonFields.CASES] == 2
    assert cases_by_level.has_url.at[AggregationLevel.COUNTY.value, CommonFields.CASES] == 1
    assert (
        cases_by_level.annotation_count.at[AggregationLevel.COUNTY.value, CommonFields.CASES] == 2
    )

    cases_by_group = per_region.subset_variables([CommonFields.CASES]).aggregate(
        timeseries_stats.RegionAggregationMethod.LEVEL,
        timeseries_stats.VariableAggregationMethod.FIELD_GROUP,
    )
    assert (
        cases_by_group.has_timeseries.at[AggregationLevel.COUNTY.value, FieldGroup.CASES_DEATHS]
        == 2
    )
    assert cases_by_group.has_url.at[AggregationLevel.COUNTY.value, FieldGroup.CASES_DEATHS] == 1
    assert (
        cases_by_group.annotation_count.at[AggregationLevel.COUNTY.value, FieldGroup.CASES_DEATHS]
        == 2
    )

    per_region.stats_for_locations(dataset.location_ids)

    counties = dataset.get_subset(aggregation_level=AggregationLevel.COUNTY)
    county_stats = timeseries_stats.PerTimeseriesStats.make(counties).aggregate_buckets()
    pop_by_var_has_url = population_ratio_by_variable(counties, county_stats.has_url)
    assert not pop_by_var_has_url.empty
    pop_by_var_has_ts = population_ratio_by_variable(counties, county_stats.has_timeseries)
    assert not pop_by_var_has_ts.empty
