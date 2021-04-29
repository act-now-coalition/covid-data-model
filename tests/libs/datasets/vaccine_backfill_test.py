import pytest

from covidactnow.datapublic.common_fields import CommonFields
from covidactnow.datapublic.common_fields import DemographicBucket

from libs.datasets.taglib import TagType
from libs.pipeline import Region
from tests import test_helpers
from tests.test_helpers import TimeseriesLiteral
from libs.datasets import vaccine_backfills


@pytest.mark.parametrize(
    "initiated_values,initiated_expected,annotation",
    [([50, None], [50, None], False), ([None, None], [50, 150], True)],
)
def test_backfill_vaccine_initiated(initiated_values, initiated_expected, annotation):
    ny_region = Region.from_state("NY")
    az_region = Region.from_state("AZ")

    # Initiated has a hole, but since they're reporting some data we don't
    # want to fill
    ny_metrics = {
        CommonFields.VACCINES_ADMINISTERED: [100, 200],
        CommonFields.VACCINATIONS_INITIATED: initiated_values,
        CommonFields.VACCINATIONS_COMPLETED: [50, 50],
    }
    az_metrics = {
        CommonFields.VACCINES_ADMINISTERED: [100, 200],
        CommonFields.VACCINATIONS_COMPLETED: [30, 40],
        # These are intentionally incorrect to make sure the computed initiated is dropped.
        CommonFields.VACCINATIONS_INITIATED: [71, 161],
    }
    metrics = {ny_region: ny_metrics, az_region: az_metrics}
    dataset = test_helpers.build_dataset(metrics)
    result = vaccine_backfills.backfill_vaccination_initiated(dataset)
    derived = test_helpers.make_tag(TagType.DERIVED, f="backfill_vaccination_initiated")
    if annotation:
        initiated_expected = TimeseriesLiteral(initiated_expected, annotation=[derived])
    expected_ny = {
        CommonFields.VACCINES_ADMINISTERED: [100, 200],
        CommonFields.VACCINATIONS_COMPLETED: [50, 50],
        CommonFields.VACCINATIONS_INITIATED: initiated_expected,
    }
    expected_metrics = {ny_region: expected_ny, az_region: az_metrics}
    expected_dataset = test_helpers.build_dataset(expected_metrics)

    test_helpers.assert_dataset_like(result, expected_dataset)


def test_backfill_vaccine_initiated_by_bucket():
    bucket_all = DemographicBucket.ALL
    bucket_40s = DemographicBucket("age:40-49")
    derived = test_helpers.make_tag(TagType.DERIVED, f="backfill_vaccination_initiated")

    ds_in = test_helpers.build_default_region_dataset(
        {
            CommonFields.VACCINES_ADMINISTERED: {bucket_all: [100, 200], bucket_40s: [40, 60]},
            CommonFields.VACCINATIONS_INITIATED: [None, None],
            CommonFields.VACCINATIONS_COMPLETED: {bucket_all: [50, 50], bucket_40s: [10, 20]},
        }
    )

    ds_result = vaccine_backfills.backfill_vaccination_initiated(ds_in)
    ds_expected = test_helpers.build_default_region_dataset(
        {
            CommonFields.VACCINES_ADMINISTERED: {bucket_all: [100, 200], bucket_40s: [40, 60]},
            CommonFields.VACCINATIONS_INITIATED: {
                bucket_all: TimeseriesLiteral([50, 150], annotation=[derived]),
                bucket_40s: TimeseriesLiteral([30, 40], annotation=[derived]),
            },
            CommonFields.VACCINATIONS_COMPLETED: {bucket_all: [50, 50], bucket_40s: [10, 20]},
        }
    )

    test_helpers.assert_dataset_like(ds_result, ds_expected)


def test_backfill_vaccine_without_completed():
    """Make sure nothing is changed when VACCINATIONS_COMPLETED is incomplete."""
    region_tx = Region.from_state("TX")
    region_sf = Region.from_fips("06075")
    metrics_tx = {
        CommonFields.VACCINES_ADMINISTERED: [None, 200],
        CommonFields.VACCINATIONS_COMPLETED: [200, None],
    }
    metrics_sf = {
        CommonFields.VACCINES_ADMINISTERED: [300, 300],
        CommonFields.VACCINATIONS_COMPLETED: [100, 100],
    }
    ds_in = test_helpers.build_dataset({region_tx: metrics_tx, region_sf: metrics_sf})

    ds_result = vaccine_backfills.backfill_vaccination_initiated(ds_in)

    derived = test_helpers.make_tag(TagType.DERIVED, f="backfill_vaccination_initiated")
    ds_expected = test_helpers.build_dataset(
        {
            region_tx: metrics_tx,
            region_sf: {
                **metrics_sf,
                CommonFields.VACCINATIONS_INITIATED: TimeseriesLiteral(
                    [200, 200], annotation=[derived]
                ),
            },
        }
    )

    test_helpers.assert_dataset_like(ds_result, ds_expected)


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
