import dataclasses
import pytest
from covidactnow.datapublic.common_fields import CommonFields
from covidactnow.datapublic.common_fields import FieldName

from libs import pipeline
from libs.datasets import combined_datasets
from libs.datasets import custom_aggregations
from libs.pipeline import Region
from tests import test_helpers


@pytest.mark.slow
def test_aggregate_to_new_york_city(nyc_region):
    dataset_in = combined_datasets.load_us_timeseries_dataset().get_regions_subset(
        custom_aggregations.ALL_NYC_REGIONS
    )
    dataset_out = custom_aggregations.aggregate_to_new_york_city(dataset_in)
    assert dataset_out


@pytest.mark.slow
def test_replace_dc_county(nyc_region):
    dc_state_region = pipeline.Region.from_fips("11")
    dc_county_region = pipeline.Region.from_fips("11001")

    dataset = combined_datasets.load_us_timeseries_dataset().get_regions_subset(
        [nyc_region, dc_state_region, dc_county_region]
    )
    # TODO(tom): Find some way to have this test read data that hasn't already gone
    # through replace_dc_county_with_state_data and remove this hack.
    timeseries_modified = dataset.timeseries.copy()
    dc_state_rows = (
        timeseries_modified.index.get_level_values("location_id") == dc_state_region.location_id
    )
    # Modify DC state cases so that they're not equal to county values.
    timeseries_modified.loc[dc_state_rows, CommonFields.CASES] = 10
    dataset = dataclasses.replace(dataset, timeseries=timeseries_modified, timeseries_bucketed=None)

    # Verify that county and state DC numbers are different to better
    # assert after that the replace worked.
    dc_state_dataset = dataset.get_regions_subset([dc_state_region])
    dc_county_dataset = dataset.get_regions_subset([dc_county_region])
    ts1 = dc_state_dataset.timeseries[CommonFields.CASES].reset_index(drop=True)
    ts2 = dc_county_dataset.timeseries[CommonFields.CASES].reset_index(drop=True)
    assert not ts1.equals(ts2)

    output = custom_aggregations.replace_dc_county_with_state_data(dataset)

    # Verify that the regions are the same input and output
    assert dataset.timeseries_regions == output.timeseries_regions

    # Verify that the state cases from before replacement are now the
    # same as the county cases
    dc_county_dataset = output.get_regions_subset([dc_county_region])
    ts1 = dc_state_dataset.timeseries[CommonFields.CASES].reset_index(drop=True)
    ts2 = dc_county_dataset.timeseries[CommonFields.CASES].reset_index(drop=True)
    assert ts1.equals(ts2)


def test_calculate_puerto_rico_bed_occupancy_rate():
    # TODO(tom): Test ALL_BED_TYPICAL_OCCUPANCY_RATE and ICU_TYPICAL_OCCUPANCY_RATE aggregation.
    field_already_agg = FieldName("already_aggregated")
    field_to_agg = FieldName("to_aggregate")
    field_other = FieldName("other")

    region_pr_state = Region.from_state("PR")
    region_jd = Region.from_fips("72075")
    region_rg = Region.from_fips("72119")
    region_la = Region.from_fips("06037")

    ts_data = {region_pr_state: {field_other: [7, 8]}, region_la: {field_other: [70, 80]}}
    static_pr_state = {
        field_already_agg: 10,
    }
    static_others = {
        region_jd: {field_to_agg: 2, field_already_agg: 4},
        region_rg: {field_to_agg: 3},
        region_la: {field_to_agg: 100, field_already_agg: 200},
    }
    ds_in = test_helpers.build_dataset(
        ts_data,
        static_by_region_then_field_name={region_pr_state: static_pr_state, **static_others},
    )

    ds_out = custom_aggregations.aggregate_puerto_rico_from_counties(ds_in)

    ds_expected = test_helpers.build_dataset(
        ts_data,
        static_by_region_then_field_name={
            region_pr_state: {field_to_agg: 5, **static_pr_state},
            **static_others,
        },
    )

    test_helpers.assert_dataset_like(ds_out, ds_expected)
