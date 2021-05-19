from covidactnow.datapublic.common_fields import CommonFields

from libs.datasets.taglib import TagType
from libs.pipeline import Region
from tests import test_helpers
from tests.test_helpers import TimeseriesLiteral
from libs.datasets import vaccine_backfills


def test_derive_vaccine_pct():
    region_tx = Region.from_state("TX")
    region_sf = Region.from_fips("06075")
    # TX has metrics that will be transformed to a percent of the population. Include some other
    # data to make sure it is not dropped.
    tx_timeseries_in = {
        CommonFields.VACCINATIONS_INITIATED: TimeseriesLiteral(
            [1_000, 2_000],
            annotation=[test_helpers.make_tag(date="2020-04-01")],
            provenance=["prov1"],
        ),
        CommonFields.VACCINATIONS_COMPLETED: [None, 1_000],
        CommonFields.CASES: TimeseriesLiteral([1, 2], provenance=["caseprov"]),
    }
    # SF does not have any vaccination metrics that can be transformed to a percentage so for it
    # the input and output are the same.
    sf_timeseries = {
        CommonFields.VACCINATIONS_COMPLETED_PCT: [0.1, 1],
    }
    static_data_map = {
        region_tx: {CommonFields.POPULATION: 100_000},
        region_sf: {CommonFields.POPULATION: 10_000},
    }

    ds_in = test_helpers.build_dataset(
        {region_tx: tx_timeseries_in, region_sf: sf_timeseries},
        static_by_region_then_field_name=static_data_map,
    )

    ds_out = vaccine_backfills.derive_vaccine_pct(ds_in)

    ds_expected = test_helpers.build_dataset(
        {
            region_tx: {
                **tx_timeseries_in,
                CommonFields.VACCINATIONS_INITIATED_PCT: [1, 2],
                CommonFields.VACCINATIONS_COMPLETED_PCT: [None, 1],
            },
            region_sf: sf_timeseries,
        },
        static_by_region_then_field_name=static_data_map,
    )

    test_helpers.assert_dataset_like(ds_out, ds_expected)


def test_derive_vaccine_pct_least_stale():
    # Test data with cumulative people counts and percentages having slightly different values so
    # that they can be differentiated.
    ts_metrics_in = {
        # These two have the same freshness so the percentage is left untouched.
        CommonFields.VACCINATIONS_INITIATED: [500, 600, 700],
        CommonFields.VACCINATIONS_INITIATED_PCT: [51, 61, 71],
        # The completed percentage is less fresh (NA for most recent date) so will be overwritten
        # by a time series derived from the cumulative people count.
        CommonFields.VACCINATIONS_COMPLETED: [700, 800, 900],
        CommonFields.VACCINATIONS_COMPLETED_PCT: [71, 81, None],
    }
    static = {CommonFields.POPULATION: 1_000}
    ds_in = test_helpers.build_default_region_dataset(ts_metrics_in, static=static)

    ds_out = vaccine_backfills.derive_vaccine_pct(ds_in)

    ts_metrics_expected = {**ts_metrics_in, CommonFields.VACCINATIONS_COMPLETED_PCT: [70, 80, 90]}
    ds_expected = test_helpers.build_default_region_dataset(ts_metrics_expected, static=static)

    test_helpers.assert_dataset_like(ds_out, ds_expected)


def test_estimate_initiated_from_state_ratio():
    region_no = Region.from_fips("22071")  # Orleans Parish
    metrics_unmodified_regions = {
        # State of Louisiana, used to estimate region_no vaccinations
        Region.from_state("LA"): {
            CommonFields.VACCINATIONS_COMPLETED: [100, 200],
            CommonFields.VACCINATIONS_INITIATED: [50, 150],
        },
    }

    ds_in = test_helpers.build_dataset(
        {**metrics_unmodified_regions, region_no: {CommonFields.VACCINATIONS_COMPLETED: [20, 40]}}
    )

    ds_result = vaccine_backfills.estimate_initiated_from_state_ratio(ds_in)
    derived = test_helpers.make_tag(
        TagType.DERIVED, function_name="estimate_initiated_from_state_ratio"
    )
    ds_expected = test_helpers.build_dataset(
        {
            **metrics_unmodified_regions,
            region_no: {
                CommonFields.VACCINATIONS_COMPLETED: [20, 40],
                CommonFields.VACCINATIONS_INITIATED: TimeseriesLiteral(
                    [10, 30], annotation=[derived]
                ),
            },
        }
    )
    test_helpers.assert_dataset_like(ds_result, ds_expected)
