import logging
import json
import pathlib
from typing import Mapping
from typing import Optional

import click

import us
from datapublic.common_fields import CommonFields
from datapublic.common_fields import PdFields

import api
import pyseir.cli
import pyseir.run

from api import update_open_api_spec
from api import data_overview_builder
from libs.datasets import timeseries
from libs.datasets.timeseries import MultiRegionDataset
from libs.metrics import test_positivity
from libs.datasets import combined_datasets
from libs.datasets.dataset_utils import REPO_ROOT
from libs.datasets.dataset_utils import AggregationLevel
import pyseir.utils
import pandas as pd

from libs.pipelines import api_v2_pipeline

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
    "--data-overview-path",
    "-d",
    type=pathlib.Path,
    help="Path of data overview documentation page.",
    default="api/docs/docs/data-definitions.md",
)
@click.option(
    "--schemas-output-dir",
    "-s",
    type=pathlib.Path,
    help="Output directory json-schema outputs.",
    default="api/schemas_v2/",
)
def update_schemas(api_output_path, data_overview_path, schemas_output_dir):
    """Updates all public facing API schemas."""
    spec = update_open_api_spec.construct_open_api_spec()
    api_output_path.write_text(json.dumps(spec, indent=2))

    schemas = api.find_public_model_classes(api_v2=True)
    for schema in schemas:
        path = schemas_output_dir / f"{schema.__name__}.json"
        _logger.info(f"Updating schema {schema} to {path}")
        path.write_text(schema.schema_json(indent=2))

    _logger.info(f"Updating data overview page -> {data_overview_path}")
    output = data_overview_builder.build_sections()
    data_overview_path.write_text(output)


@main.command()
@click.option(
    "--test-positivity-all-methods", default="test-positivity-all.csv", type=pathlib.Path,
)
@click.option(
    "--final-result", default="test-positivity-output.csv", type=pathlib.Path,
)
@click.option(
    "--output-dir", default="results/output", type=pathlib.Path,
)
@click.option("--state", type=str)
@click.option("--fips", type=str)
def generate_test_positivity(
    test_positivity_all_methods: pathlib.Path,
    final_result: pathlib.Path,
    output_dir: pathlib.Path,
    state: Optional[str],
    fips: Optional[str],
):
    if state:
        active_states = [state]
    else:
        active_states = [state.abbr for state in us.STATES]
        active_states = active_states + ["PR", "MP"]

    selected_dataset = combined_datasets.load_us_timeseries_dataset().get_subset(
        exclude_county_999=True, states=active_states, fips=fips,
    )
    test_positivity_results = test_positivity.AllMethods.run(selected_dataset)
    _write_dataset_map(
        output_dir / test_positivity_all_methods, test_positivity_results.all_methods_datasets
    )

    test_positivity_results.test_positivity.timeseries_rows().to_csv(
        output_dir / final_result, index=True, float_format="%.05g"
    )


def _write_dataset_map(
    output_path: pathlib.Path,
    all_methods_datasets: Mapping[timeseries.DatasetName, MultiRegionDataset],
):
    """Writes a map from DatasetName to dataset, used for debugging test positivity."""
    all_datasets_df = pd.concat(
        {name: ds.timeseries_rows() for name, ds in all_methods_datasets.items()},
        names=[PdFields.DATASET, CommonFields.LOCATION_ID, PdFields.VARIABLE],
    )
    all_methods = (
        all_datasets_df.xs(CommonFields.TEST_POSITIVITY, level=PdFields.VARIABLE)
        .sort_index()
        .reorder_levels([CommonFields.LOCATION_ID, PdFields.DATASET])
    )
    all_methods.sort_index().to_csv(output_path, index=True, float_format="%.05g")


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
    _logger.info(f"Loading all API Regions.")
    selected_dataset = combined_datasets.load_us_timeseries_dataset().get_subset(
        aggregation_level=level, exclude_county_999=True, state=state, fips=fips,
    )

    _logger.info(f"Loading pyseir results.")
    model_output = pyseir.run.PyseirOutputDatasets.read(model_output_dir)

    _logger.info(f"Running generate_from_loaded_data")
    api_v2_pipeline.generate_from_loaded_data(model_output, output, selected_dataset, _logger)
