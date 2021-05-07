import pytest
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

    ds_in = test_helpers.build_dataset(
        {
            **states_timeseries,
            # region_us input intentionally not the sum of states to check that the input is preserved.
            region_us: {CommonFields.CASES: [2, 6]},
        },
        static_by_region_then_field_name=states_static,
    )
    ds_out = data.run_aggregate_to_country(ds_in)

    ds_expected = test_helpers.build_dataset(
        {
            **states_timeseries,
            # region_us input intentionally not the sum of states to check that the input is preserved.
            region_us: {
                CommonFields.CASES: [2, 6],
                CommonFields.ALL_BED_TYPICAL_OCCUPANCY_RATE: [0.6, 0.6],
            },
        },
        static_by_region_then_field_name={
            **states_static,
            **{
                region_us: {
                    CommonFields.POPULATION: 200,
                    CommonFields.MAX_BED_COUNT: 20,
                    CommonFields.ICU_BEDS: 4,
                }
            },
        },
    )

    test_helpers.assert_dataset_like(ds_out, ds_expected)
