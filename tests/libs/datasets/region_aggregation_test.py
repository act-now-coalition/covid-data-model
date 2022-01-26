import io

from datapublic.common_fields import CommonFields
from datapublic.common_fields import FieldName

from libs.datasets import region_aggregation

from libs.datasets import timeseries
from libs.pipeline import Region
from tests import test_helpers


def test_aggregate_states_to_country():
    ts = timeseries.MultiRegionDataset.from_csv(
        io.StringIO(
            "location_id,county,aggregate_level,date,m1,m2,population\n"
            "iso1:us#fips:97111,Bar County,county,2020-04-03,3,,\n"
            "iso1:us#fips:97222,Foo County,county,2020-04-01,,10,\n"
            "iso1:us#iso2:us-tx,Texas,state,2020-04-01,1,2,\n"
            "iso1:us#iso2:us-tx,Texas,state,2020-04-02,3,4,\n"
            "iso1:us#iso2:us-tx,Texas,state,,,,1000\n"
            "iso1:us#iso2:us-az,Arizona,state,2020-04-01,1,2,\n"
            "iso1:us#iso2:us-az,Arizona,state,,,,2000\n"
        )
    )
    region_us = Region.from_iso1("us")
    country = region_aggregation.aggregate_regions(
        ts,
        {Region.from_state("AZ"): region_us, Region.from_state("TX"): region_us},
        [],
        reporting_ratio_required_to_aggregate=1.0,
    )
    expected = timeseries.MultiRegionDataset.from_csv(
        io.StringIO(
            "location_id,aggregate_level,date,m1,m2,population\n"
            "iso1:us,country,2020-04-01,2,4,\n"
            "iso1:us,country,,,,3000\n"
        )
    )
    test_helpers.assert_dataset_like(country, expected)


def test_aggregate_states_to_country_scale():
    ts = timeseries.MultiRegionDataset.from_csv(
        io.StringIO(
            "location_id,county,aggregate_level,date,m1,m2,population\n"
            "iso1:us#iso2:us-tx,Texas,state,2020-04-01,4,2,\n"
            "iso1:us#iso2:us-tx,Texas,state,2020-04-02,4,4,\n"
            "iso1:us#iso2:us-tx,Texas,state,,,,2500\n"
            "iso1:us#iso2:us-az,Arizona,state,2020-04-01,8,20,\n"
            "iso1:us#iso2:us-az,Arizona,state,2020-04-02,12,40,\n"
            "iso1:us#iso2:us-az,Arizona,state,,,,7500\n"
        )
    )
    region_us = Region.from_iso1("us")
    country = region_aggregation.aggregate_regions(
        ts,
        {Region.from_state("AZ"): region_us, Region.from_state("TX"): region_us},
        [
            region_aggregation.StaticWeightedAverageAggregation(
                FieldName("m1"), CommonFields.POPULATION
            ),
        ],
    )
    # The column m1 is scaled by population.
    # On 2020-04-01: 4 * 0.25 + 8 * 0.75 = 7
    # On 2020-04-02: 4 * 0.25 + 12 * 0.75 = 10
    expected = timeseries.MultiRegionDataset.from_csv(
        io.StringIO(
            "location_id,aggregate_level,date,m1,m2,population\n"
            "iso1:us,country,2020-04-01,7,22,\n"
            "iso1:us,country,2020-04-02,10,44,\n"
            "iso1:us,country,,,,10000\n"
        )
    )
    test_helpers.assert_dataset_like(country, expected)


def test_aggregate_states_to_country_scale_static():
    ts = timeseries.MultiRegionDataset.from_csv(
        io.StringIO(
            "location_id,county,aggregate_level,date,m1,s1,population\n"
            "iso1:us#iso2:us-tx,Texas,state,2020-04-01,4,,\n"
            "iso1:us#iso2:us-tx,Texas,state,,,4,2500\n"
            "iso1:us#iso2:us-az,Arizona,state,2020-04-01,8,,\n"
            "iso1:us#iso2:us-az,Arizona,state,,,12,7500\n"
        )
    )
    region_us = Region.from_iso1("us")
    country = region_aggregation.aggregate_regions(
        ts,
        {Region.from_state("AZ"): region_us, Region.from_state("TX"): region_us},
        [
            region_aggregation.StaticWeightedAverageAggregation(
                FieldName("m1"), CommonFields.POPULATION
            ),
            region_aggregation.StaticWeightedAverageAggregation(
                FieldName("s1"), CommonFields.POPULATION
            ),
        ],
    )
    # The column m1 is scaled by population.
    # 4 * 0.25 + 12 * 0.75 = 10
    expected = timeseries.MultiRegionDataset.from_csv(
        io.StringIO(
            "location_id,aggregate_level,date,m1,s1,population\n"
            "iso1:us,country,2020-04-01,7,,\n"
            "iso1:us,country,,,10,10000\n"
        )
    )
    test_helpers.assert_dataset_like(country, expected)
