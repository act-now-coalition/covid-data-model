import logging
import json
import pathlib
from typing import Optional

import click
import structlog

import us
from covidactnow.datapublic.common_fields import CommonFields
from covidactnow.datapublic.common_fields import PdFields

import api

from api import update_open_api_spec
from libs import test_positivity
from libs import top_level_metrics
from libs.pipelines import api_v2_pipeline
from libs.datasets import combined_datasets
from libs.datasets.timeseries import MultiRegionDataset
from libs.datasets.dataset_utils import REPO_ROOT
from libs.datasets.dataset_utils import AggregationLevel
from pyseir.utils import SummaryArtifact
import pandas as pd

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
    test_positivity_results.write(output_dir / test_positivity_all_methods)

    # Similar to test_positivity.run_and_maybe_join_columns
    joined_dataset = selected_dataset.drop_column_if_present(
        CommonFields.TEST_POSITIVITY
    ).join_columns(test_positivity_results.test_positivity)

    log = structlog.get_logger()
    positivity_time_series = {}
    source_map = {}
    for region, regional_data in joined_dataset.iter_one_regions():
        pos_series, details = top_level_metrics.calculate_or_copy_test_positivity(
            regional_data, log
        )
        positivity_time_series[region.location_id] = pos_series
        source_map[region.location_id] = details.source.value
    positivity_wide_date_df = pd.DataFrame.from_dict(positivity_time_series, orient="index")
    source_df = pd.DataFrame.from_dict(source_map, orient="index", columns=[PdFields.PROVENANCE])
    df = pd.concat([source_df, positivity_wide_date_df], axis=1, sort=True)
    # The column headers are output as yyyy-mm-dd 00:00:00; I haven't found an easy way to write
    # only the date.
    df.to_csv(output_dir / final_result, index=True, float_format="%.05g")


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
