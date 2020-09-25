import logging
import pathlib
from typing import Optional

import click
import us

import pydantic
import api
from api import update_open_api_spec
from api.can_api_definition import RegionSummaryWithTimeseries
from api.can_api_definition import AggregateRegionSummaryWithTimeseries
from libs import update_readme_schemas
from libs.pipelines import api_pipeline
from libs.pipelines import api_v2_pipeline
from libs.datasets import combined_datasets
from libs.datasets.timeseries import MultiRegionTimeseriesDataset
from libs.datasets.dataset_utils import REPO_ROOT
from libs.datasets.dataset_utils import AggregationLevel
from libs.enums import Intervention
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
    "--output-dir",
    "-o",
    type=pathlib.Path,
    help="Output directory to save schemas in.",
    default="api/schemas",
)
@click.option(
    "--update-readme/--skip-update-readme",
    type=bool,
    help="If true, updates readme with schemas",
    default=True,
)
def update_schemas(output_dir, update_readme):
    """Updates all public facing API schemas."""
    schemas = api.find_public_model_classes()
    for schema in schemas:
        path = output_dir / f"{schema.__name__}.json"
        _logger.info(f"Updating schema {schema} to {path}")
        path.write_text(schema.schema_json(indent=2))

    if update_readme:
        _logger.info(f"Updating {API_README_PATH} with schema definitions")
        # Generate a single schema with all API schema definitions.
        can_schema = pydantic.schema.schema(schemas)
        schemas = update_readme_schemas.generate_markdown_for_schema_definitions(can_schema)
        update_readme_schemas.generate_api_readme_from_template(
            API_README_TEMPLATE_PATH, API_README_PATH, schemas
        )


@main.command()
@click.option(
    "--api-output-path",
    "-o",
    type=pathlib.Path,
    help="Path to open api schema path to save to.",
    default="api/docs/open_api_schema.json",
)
def update_v2_schemas(api_output_path):
    """Updates all public facing API schemas."""
    spec = update_open_api_spec.construct_open_api_spec()
    schema_out = spec.json(by_alias=True, exclude_none=True, indent=2)
    api_output_path.write_text(schema_out)


@main.command()
@click.option(
    "--input-dir",
    "-i",
    default="results",
    help="Input directory of state projections",
    type=pathlib.Path,
)
@click.option(
    "--output",
    "-o",
    default="results/output/states",
    help="Output directory for artifacts",
    type=pathlib.Path,
)
@click.option(
    "--summary-output",
    default="results/output",
    help="Output directory for state summaries.",
    type=pathlib.Path,
)
@click.option("--aggregation-level", "-l", type=AggregationLevel)
@click.option("--state")
@click.option("--fips")
def generate_api(input_dir, output, summary_output, aggregation_level, state, fips):
    """The entry function for invocation"""

    active_states = [state.abbr for state in us.STATES]
    active_states = active_states + ["PR", "MP"]
    regions = combined_datasets.get_subset_regions(
        aggregation_level=aggregation_level,
        exclude_county_999=True,
        state=state,
        fips=fips,
        states=active_states,
    )

    icu_data_path = input_dir / SummaryArtifact.ICU_METRIC_COMBINED.value
    icu_data = MultiRegionTimeseriesDataset.from_csv(icu_data_path)
    rt_data_path = input_dir / SummaryArtifact.RT_METRIC_COMBINED.value
    rt_data = MultiRegionTimeseriesDataset.from_csv(rt_data_path)

    for intervention in list(Intervention):
        _logger.info(f"Running intervention {intervention.name}")
        regional_inputs = [
            api_pipeline.RegionalInput.from_region_and_intervention(
                region, intervention, input_dir, rt_data, icu_data
            )
            for region in regions
        ]
        _logger.info(f"Loaded {len(regional_inputs)} regions.")
        all_timeseries = api_pipeline.run_on_all_regional_inputs_for_intervention(regional_inputs)
        county_timeseries = [
            output for output in all_timeseries if output.aggregate_level is AggregationLevel.COUNTY
        ]
        api_pipeline.deploy_single_level(intervention, county_timeseries, summary_output, output)
        state_timeseries = [
            output for output in all_timeseries if output.aggregate_level is AggregationLevel.STATE
        ]
        api_pipeline.deploy_single_level(intervention, state_timeseries, summary_output, output)


@main.command()
@click.argument("model-output-dir", type=pathlib.Path)
@click.option(
    "--output",
    "-o",
    default="results/output/states",
    help="Output directory for artifacts",
    type=pathlib.Path,
)
@click.option("--aggregation-level", "-l", type=AggregationLevel)
@click.option("--state")
@click.option("--fips")
def generate_api_v2(model_output_dir, output, aggregation_level, state, fips):
    """The entry function for invocation"""

    active_states = [state.abbr for state in us.STATES]
    active_states = active_states + ["PR", "MP"]

    # Load all API Regions
    regions = combined_datasets.get_subset_regions(
        aggregation_level=aggregation_level,
        exclude_county_999=True,
        state=state,
        fips=fips,
        states=active_states,
    )
    _logger.info(f"Loading all regional inputs.")
    icu_data_path = model_output_dir / SummaryArtifact.ICU_METRIC_COMBINED.value
    icu_data = MultiRegionTimeseriesDataset.from_csv(icu_data_path)
    rt_data_path = model_output_dir / SummaryArtifact.RT_METRIC_COMBINED.value
    rt_data = MultiRegionTimeseriesDataset.from_csv(rt_data_path)
    regional_inputs = [
        api_v2_pipeline.RegionalInput.from_region_and_model_output(region, rt_data, icu_data)
        for region in regions
    ]
    _logger.info(f"Finished loading all regional inputs.")

    # Build all region timeseries API Output objects.
    _logger.info("Generating all API Timeseries")
    all_timeseries = api_v2_pipeline.run_on_regions(regional_inputs)

    api_v2_pipeline.deploy_single_level(all_timeseries, AggregationLevel.COUNTY, output)
    api_v2_pipeline.deploy_single_level(all_timeseries, AggregationLevel.STATE, output)

    _logger.info("Finished API generation.")


@main.command("generate-top-counties")
@click.option(
    "--disable-validation", "-dv", is_flag=True, help="Run the validation on the deploy command"
)
@click.option(
    "--input-dir", "-i", default="results", help="Input directory of state/county projections"
)
@click.option(
    "--output",
    "-o",
    default="results/top_counties",
    help="Output directory for artifacts",
    type=pathlib.Path,
)
@click.option("--state")
@click.option("--fips")
def generate_top_counties(disable_validation, input_dir, output, state, fips: Optional[str]):
    """The entry function for invocation"""
    intervention = Intervention.SELECTED_INTERVENTION
    active_states = [state.abbr for state in us.STATES] + ["PR"]
    regions = combined_datasets.get_subset_regions(
        aggregation_level=AggregationLevel.COUNTY,
        exclude_county_999=True,
        states=active_states,
        state=state,
        fips=fips,
    )
    regional_inputs = [
        api_pipeline.RegionalInput.from_region_and_intervention(region, intervention, input_dir)
        for region in regions
    ]

    def sort_func(output: RegionSummaryWithTimeseries):
        return -output.projections.totalHospitalBeds.peakShortfall

    all_timeseries = api_pipeline.run_on_all_regional_inputs_for_intervention(
        regional_inputs, sort_func=sort_func, limit=100,
    )
    bulk_timeseries = AggregateRegionSummaryWithTimeseries(__root__=all_timeseries)

    api_pipeline.deploy_json_api_output(
        intervention, bulk_timeseries, output, filename_override="counties_top_100.json"
    )
    _logger.info("Finished top counties job")
