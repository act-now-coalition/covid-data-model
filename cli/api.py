import logging
import json
import pathlib
import click

import us

import pydantic
import api
from api import update_open_api_spec
from libs import test_positivity
from libs import update_readme_schemas
from libs.pipelines import api_v2_pipeline
from libs.datasets import combined_datasets
from libs.datasets.timeseries import MultiRegionDataset
from libs.datasets.dataset_utils import REPO_ROOT
from libs.datasets.dataset_utils import AggregationLevel
from pyseir.utils import SummaryArtifact

PROD_BUCKET = "data.covidactnow.org"

API_README_TEMPLATE_PATH = REPO_ROOT / "api" / "README.V1.tmpl.md"
API_README_PATH = REPO_ROOT / "api" / "README.V1.md"


_logger = logging.getLogger(__name__)


@click.group("api")
def main():
    pass


@main.command()
@click.option(
    "--api-output-path",
    "-o",
    type=pathlib.Path,
    help="Path to open api schema path to save to.",
    default="api/docs/open_api_schema.json",
)
@click.option(
    "--schemas-output-dir",
    "-s",
    type=pathlib.Path,
    help="Output directory json-schema outputs.",
    default="api/schemas_v2/",
)
def update_v2_schemas(api_output_path, schemas_output_dir):
    """Updates all public facing API schemas."""
    spec = update_open_api_spec.construct_open_api_spec()
    api_output_path.write_text(json.dumps(spec, indent=2))

    schemas = api.find_public_model_classes(api_v2=True)
    for schema in schemas:
        path = schemas_output_dir / f"{schema.__name__}.json"
        _logger.info(f"Updating schema {schema} to {path}")
        path.write_text(schema.schema_json(indent=2))


@main.command()
@click.option(
    "--test-positivity-all-methods",
    default="results/output/test-positivity.csv",
    type=pathlib.Path,
)
def generate_test_positivity(test_positivity_all_methods: pathlib.Path):
    active_states = [state.abbr for state in us.STATES]
    active_states = active_states + ["PR", "MP"]
    selected_dataset = combined_datasets.load_us_timeseries_dataset().get_subset(
        exclude_county_999=True, states=active_states
    )
    test_positivity_results = test_positivity.AllMethods.run(selected_dataset)
    test_positivity_results.write(test_positivity_all_methods)


@main.command()
@click.argument("model-output-dir", type=pathlib.Path)
@click.option(
    "--output",
    "-o",
    default="results/output/states",
    help="Output directory for artifacts",
    type=pathlib.Path,
)
@click.option("--level", "-l", type=AggregationLevel)
@click.option("--state")
@click.option("--fips")
def generate_api_v2(model_output_dir, output, level, state, fips):
    """The entry function for invocation"""

    # Load all API Regions
    selected_dataset = combined_datasets.load_us_timeseries_dataset().get_subset(
        aggregation_level=level, exclude_county_999=True, state=state, fips=fips,
    )
    _logger.info(f"Loading all regional inputs.")

    icu_data_path = model_output_dir / SummaryArtifact.ICU_METRIC_COMBINED.value
    icu_data = MultiRegionDataset.from_csv(icu_data_path)
    icu_data_map = dict(icu_data.iter_one_regions())

    rt_data_path = model_output_dir / SummaryArtifact.RT_METRIC_COMBINED.value
    rt_data = MultiRegionDataset.from_csv(rt_data_path)
    rt_data_map = dict(rt_data.iter_one_regions())

    # If calculating test positivity succeeds join it with the combined_datasets into one
    # MultiRegionDataset
    regions_data = test_positivity.run_and_maybe_join_columns(selected_dataset, _logger)

    regional_inputs = [
        api_v2_pipeline.RegionalInput.from_one_regions(
            region,
            regional_data,
            icu_data=icu_data_map.get(region),
            rt_data=rt_data_map.get(region),
        )
        for region, regional_data in regions_data.iter_one_regions()
    ]

    _logger.info(f"Finished loading all regional inputs.")

    # Build all region timeseries API Output objects.
    _logger.info("Generating all API Timeseries")
    all_timeseries = api_v2_pipeline.run_on_regions(regional_inputs)

    api_v2_pipeline.deploy_single_level(all_timeseries, AggregationLevel.COUNTY, output)
    api_v2_pipeline.deploy_single_level(all_timeseries, AggregationLevel.STATE, output)
    api_v2_pipeline.deploy_single_level(all_timeseries, AggregationLevel.CBSA, output)

    _logger.info("Finished API generation.")
