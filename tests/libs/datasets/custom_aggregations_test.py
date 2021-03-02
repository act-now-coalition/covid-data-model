import dataclasses
import io
import pytest
from covidactnow.datapublic.common_fields import CommonFields

from libs import pipeline
from libs.datasets import combined_datasets
from libs.datasets import custom_aggregations
from libs.datasets import timeseries
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
    dataset = dataclasses.replace(dataset, timeseries=timeseries_modified)

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


@pytest.mark.skip(reason="test not written, needs proper columns")
def test_calculate_puerto_rico_bed_occupancy_rate():
    ds = timeseries.MultiRegionDataset.from_csv(
        io.StringIO(
            "location_id,county,aggregate_level,date,population\n"
            "iso1:us#iso2:us-pr,Texas,state,2020-04-01,4,,\n"
            "iso1:us#iso2:us-pr,Texas,state,,,4,2500\n"
        )
    )

    actual = custom_aggregations.aggregate_puerto_rico_from_counties(ds)

    expected = timeseries.MultiRegionDataset.from_csv(
        io.StringIO(
            "location_id,aggregate_level,date,m1,s1,population\n"
            "iso1:us,country,2020-04-01,7,,\n"
            "iso1:us,country,,,10,10000\n"
        )
    )
    test_helpers.assert_dataset_like(actual, expected)
