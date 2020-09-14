import pytest

from libs.datasets import combined_datasets
from libs.datasets.timeseries import TimeseriesDataset
from libs.enums import Intervention
from libs.pipeline import Region
from libs.pipelines import api_pipeline

NYC_FIPS = "36061"


@pytest.mark.slow
@pytest.mark.parametrize(
    "intervention",
    [
        Intervention.OBSERVED_INTERVENTION,
        Intervention.STRONG_INTERVENTION,
        Intervention.NO_INTERVENTION,
    ],
)
def test_build_timeseries_and_summary_outputs(nyc_model_output_path, nyc_fips, intervention):

    us_latest = combined_datasets.load_us_latest_dataset()
    us_timeseries = combined_datasets.load_us_timeseries_dataset()

    timeseries = api_pipeline.build_timeseries_for_fips(
        intervention, us_latest, us_timeseries, nyc_model_output_path.parent, nyc_fips
    )

    if intervention is Intervention.NO_INTERVENTION:
        # Test data does not contain no intervention model, should not output any results.
        assert not timeseries
        return

    assert timeseries

    if intervention is Intervention.STRONG_INTERVENTION:
        assert timeseries.projections
        assert timeseries.timeseries
    elif intervention is Intervention.OBSERVED_INTERVENTION:
        assert not timeseries.projections
        assert not timeseries.timeseries


def test_build_api_output_for_intervention(nyc_fips, nyc_model_output_path, tmp_path):
    nyc_region = Region.from_fips(nyc_fips)
    county_output = tmp_path / "county"
    us_latest = combined_datasets.load_us_latest_dataset()
    us_timeseries = combined_datasets.load_us_timeseries_dataset()

    nyc_latest = us_latest.get_subset(None, fips=nyc_fips)
    nyc_timeseries = TimeseriesDataset(us_timeseries.get_one_region(nyc_region).data)
    all_timeseries_api = api_pipeline.run_on_all_fips_for_intervention(
        nyc_latest, nyc_timeseries, Intervention.STRONG_INTERVENTION, nyc_model_output_path.parent
    )

    api_pipeline.deploy_single_level(
        Intervention.STRONG_INTERVENTION, all_timeseries_api, tmp_path, county_output
    )
    expected_outputs = [
        "counties.STRONG_INTERVENTION.timeseries.json",
        "counties.STRONG_INTERVENTION.csv",
        "counties.STRONG_INTERVENTION.timeseries.csv",
        "counties.STRONG_INTERVENTION.json",
        "county/36061.STRONG_INTERVENTION.json",
        "county/36061.STRONG_INTERVENTION.timeseries.json",
    ]

    output_paths = [
        str(path.relative_to(tmp_path)) for path in tmp_path.glob("**/*") if not path.is_dir()
    ]
    assert sorted(output_paths) == sorted(expected_outputs)


def test_latest_values_no_unknown_fips(tmp_path):
    unknown_fips = "69999"
    region = Region.from_fips(unknown_fips)
    us_latest = combined_datasets.load_us_latest_dataset()
    us_timeseries = combined_datasets.load_us_timeseries_dataset()
    latest = us_latest.get_subset(fips=unknown_fips)
    timeseries = TimeseriesDataset(us_timeseries.get_one_region(region).data)
    all_timeseries_api = api_pipeline.run_on_all_fips_for_intervention(
        latest, timeseries, Intervention.OBSERVED_INTERVENTION, tmp_path
    )
    assert not all_timeseries_api


def test_output_no_timeseries_rows(nyc_fips, tmp_path):
    nyc_region = Region.from_fips(nyc_fips)
    us_latest = combined_datasets.load_us_latest_dataset()
    us_timeseries = combined_datasets.load_us_timeseries_dataset()
    latest = us_latest.get_subset(fips=nyc_fips)
    timeseries = TimeseriesDataset(us_timeseries.get_one_region(nyc_region).data)

    # Clearing out all rows, testing that a region with no rows still has an API output.
    timeseries.data = timeseries.data.loc[timeseries.data.fips.isna()]

    assert timeseries.empty

    all_timeseries_api = api_pipeline.run_on_all_fips_for_intervention(
        latest, timeseries, Intervention.OBSERVED_INTERVENTION, tmp_path
    )

    assert all_timeseries_api
