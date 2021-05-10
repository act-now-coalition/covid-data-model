import pytest
import structlog
from click.testing import CliRunner
from covidactnow.datapublic.common_fields import CommonFields

from cli import data
from libs.pipeline import Region
from tests import test_helpers


@pytest.mark.skip(reason="mysteriously crashes on server, see PR 969")
@pytest.mark.slow
def test_population_filter(tmp_path):
    runner = CliRunner()
    output_path = tmp_path / "filtered.csv"
    runner.invoke(
        data.run_population_filter, [str(output_path)], catch_exceptions=False,
    )
    assert output_path.exists()


def test_run_aggregate_to_country():
    region_tx = Region.from_state("TX")
    region_il = Region.from_state("IL")
    region_us = Region.from_iso1("us")

    # State data has some hospital and cases time series.
    states_timeseries = {
        region_il: {
            CommonFields.ALL_BED_TYPICAL_OCCUPANCY_RATE: [0.5, 0.5],
            CommonFields.CASES: [1, 2],
        },
        region_tx: {
            CommonFields.ALL_BED_TYPICAL_OCCUPANCY_RATE: [0.7, 0.7],
            CommonFields.CASES: [0, 3],
        },
    }
    states_static = {
        r: {CommonFields.POPULATION: 100, CommonFields.MAX_BED_COUNT: 10, CommonFields.ICU_BEDS: 2}
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
        ds_out = data.run_aggregate_to_country(ds_in)
    assert not logs

    ds_expected = test_helpers.build_dataset(
        {
            **states_timeseries,
            region_us: {
                **us_timeseries_in,
                CommonFields.ALL_BED_TYPICAL_OCCUPANCY_RATE: [0.6, 0.6],
            },
        },
        static_by_region_then_field_name={
            **states_static,
            region_us: {
                # All of the US static values are aggregated from states_static
                CommonFields.POPULATION: 200,
                CommonFields.MAX_BED_COUNT: 20,
                CommonFields.ICU_BEDS: 4,
            },
        },
    )

    test_helpers.assert_dataset_like(ds_out, ds_expected)


def test_run_aggregate_to_country_unexpected_unaggregated_value_is_logged():
    region_tx = Region.from_state("TX")
    region_il = Region.from_state("IL")
    region_us = Region.from_iso1("us")

    assert CommonFields.TOTAL_TESTS not in data.US_AGGREGATED_EXPECTED_VARIABLES_TO_DROP

    states_timeseries = {
        region_il: {CommonFields.TOTAL_TESTS: [1, 2]},
        region_tx: {CommonFields.TOTAL_TESTS: [0, 3]},
        # region_us input TOTAL_TESTS intentionally not the sum of states TOTAL_TESTS to check that
        # the region_us input is preserved.
        region_us: {CommonFields.TOTAL_TESTS: [2, 6]},
    }
    states_static = {
        r: {CommonFields.POPULATION: 100, CommonFields.MAX_BED_COUNT: 10, CommonFields.ICU_BEDS: 2}
        for r in [region_il, region_tx]
    }

    ds_in = test_helpers.build_dataset(
        states_timeseries, static_by_region_then_field_name=states_static
    )

    with structlog.testing.capture_logs() as logs:
        ds_out = data.run_aggregate_to_country(ds_in)
    assert [l["event"] for l in logs] == [data.US_AGGREGATED_VARIABLE_DROP_MESSAGE]

    ds_expected = test_helpers.build_dataset(
        states_timeseries,
        static_by_region_then_field_name={
            **states_static,
            region_us: {
                # All of the US static values are aggregated from states_static
                CommonFields.POPULATION: 200,
                CommonFields.MAX_BED_COUNT: 20,
                CommonFields.ICU_BEDS: 4,
            },
        },
    )

    test_helpers.assert_dataset_like(ds_out, ds_expected)
