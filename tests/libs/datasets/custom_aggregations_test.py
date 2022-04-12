import dataclasses
import pytest
import structlog
from datapublic.common_fields import CommonFields
from datapublic.common_fields import FieldName

import libs.datasets
from libs import pipeline
from libs.datasets import combined_datasets
from libs.datasets import custom_aggregations
from libs.pipeline import Region
from tests import test_helpers


@pytest.mark.slow
# @pytest.mark.skip(reason="Github action runner runs OOM when loading full dataset")
def test_aggregate_to_new_york_city(nyc_region):
    dataset_in = combined_datasets.load_us_timeseries_dataset(
        load_demographics=False
    ).get_regions_subset(custom_aggregations.ALL_NYC_REGIONS)
    dataset_out = custom_aggregations.aggregate_to_new_york_city(dataset_in)
    assert dataset_out


@pytest.mark.slow
# @pytest.mark.skip(reason="Github action runner runs OOM when loading full dataset")
def test_replace_dc_county(nyc_region):
    dc_state_region = pipeline.Region.from_fips("11")
    dc_county_region = pipeline.Region.from_fips("11001")

    dataset = combined_datasets.load_us_timeseries_dataset(
        load_demographics=False
    ).get_regions_subset([nyc_region, dc_state_region, dc_county_region])
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
    assert set(dataset.location_ids) == set(output.location_ids)

    # Verify that the state cases from before replacement are now the
    # same as the county cases
    dc_county_dataset = output.get_regions_subset([dc_county_region])
    ts1 = dc_state_dataset.timeseries[CommonFields.CASES].reset_index(drop=True)
    ts2 = dc_county_dataset.timeseries[CommonFields.CASES].reset_index(drop=True)
    assert ts1.equals(ts2)


def test_calculate_puerto_rico_bed_occupancy_rate():
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


def test_aggregate_to_country():
    region_tx = Region.from_state("TX")
    region_il = Region.from_state("IL")
    region_us = Region.from_iso1("us")

    # State data has some hospital and cases time series.
    states_timeseries = {
        region_il: {CommonFields.CURRENT_HOSPITALIZED: [5, 5], CommonFields.CASES: [1, 2],},
        region_tx: {CommonFields.CURRENT_HOSPITALIZED: [7, 7], CommonFields.CASES: [0, 3],},
    }
    states_static = {
        r: {CommonFields.POPULATION: 100, CommonFields.STAFFED_BEDS: 10, CommonFields.ICU_BEDS: 2}
        for r in [region_il, region_tx]
    }
    # region_us input CASES intentionally not the sum of states CASES to check that the
    # region_us input is not overwritten with aggregated values.
    us_timeseries_in = {CommonFields.CASES: [2, 6]}

    ds_in = test_helpers.build_dataset(
        {**states_timeseries, region_us: us_timeseries_in,},
        static_by_region_then_field_name=states_static,
    )

    with structlog.testing.capture_logs() as logs:
        ds_out = libs.datasets.custom_aggregations.aggregate_to_country(
            ds_in, reporting_ratio_required_to_aggregate=0.95
        )
    assert not logs

    ds_expected = test_helpers.build_dataset(
        {
            **states_timeseries,
            region_us: {**us_timeseries_in, CommonFields.CURRENT_HOSPITALIZED: [12, 12],},
        },
        static_by_region_then_field_name={
            **states_static,
            region_us: {
                # All of the US static values are aggregated from states_static
                CommonFields.POPULATION: 200,
                CommonFields.STAFFED_BEDS: 20,
                CommonFields.ICU_BEDS: 4,
            },
        },
    )

    test_helpers.assert_dataset_like(ds_out, ds_expected)


def test_aggregate_to_country_unexpected_unaggregated_value_is_logged():
    region_tx = Region.from_state("TX")
    region_il = Region.from_state("IL")
    region_us = Region.from_iso1("us")

    assert (
        CommonFields.TOTAL_TESTS
        not in libs.datasets.custom_aggregations.US_AGGREGATED_EXPECTED_VARIABLES_TO_DROP
    )

    states_timeseries = {
        region_il: {CommonFields.TOTAL_TESTS: [1, 2]},
        region_tx: {CommonFields.TOTAL_TESTS: [0, 3]},
        # region_us input TOTAL_TESTS intentionally not the sum of states TOTAL_TESTS to check that
        # the region_us input is preserved.
        region_us: {CommonFields.TOTAL_TESTS: [2, 6]},
    }
    states_static = {
        r: {CommonFields.POPULATION: 100, CommonFields.STAFFED_BEDS: 10, CommonFields.ICU_BEDS: 2}
        for r in [region_il, region_tx]
    }

    ds_in = test_helpers.build_dataset(
        states_timeseries, static_by_region_then_field_name=states_static
    )

    with structlog.testing.capture_logs() as logs:
        ds_out = libs.datasets.custom_aggregations.aggregate_to_country(
            ds_in, reporting_ratio_required_to_aggregate=0.95
        )
    assert [l["event"] for l in logs] == [
        libs.datasets.custom_aggregations.US_AGGREGATED_VARIABLE_DROP_MESSAGE
    ]

    ds_expected = test_helpers.build_dataset(
        states_timeseries,
        static_by_region_then_field_name={
            **states_static,
            region_us: {
                # All of the US static values are aggregated from states_static
                CommonFields.POPULATION: 200,
                CommonFields.STAFFED_BEDS: 20,
                CommonFields.ICU_BEDS: 4,
            },
        },
    )

    test_helpers.assert_dataset_like(ds_out, ds_expected)
