import logging
import pathlib
import click
import itertools
import us

import pydantic
import api
from api.can_api_definition import RegionSummaryWithTimeseries
from api.can_api_definition import AggregateRegionSummaryWithTimeseries
from libs import update_readme_schemas
from libs.pipelines import api_pipeline
from libs.datasets import combined_datasets
from libs.datasets.dataset_utils import REPO_ROOT
from libs.datasets.dataset_utils import AggregationLevel
from libs.datasets.dataset_utils import AggregationLevel
from libs.enums import Intervention

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
    active_states = active_states + ["PR"]
    us_latest = combined_datasets.load_us_latest_dataset().get_subset(
        aggregation_level, state=state, fips=fips, states=active_states
    )
    us_timeseries = combined_datasets.load_us_timeseries_dataset().get_subset(
        aggregation_level, state=state, fips=fips, states=active_states
    )

    for intervention in list(Intervention):
        _logger.info(f"Running intervention {intervention.name}")
        all_timeseries = api_pipeline.run_on_all_fips_for_intervention(
            us_latest, us_timeseries, intervention, input_dir
        )
        county_timeseries = [
            output for output in all_timeseries if output.aggregate_level is AggregationLevel.COUNTY
        ]
        api_pipeline.deploy_single_level(intervention, county_timeseries, summary_output, output)
        state_timeseries = [
            output for output in all_timeseries if output.aggregate_level is AggregationLevel.STATE
        ]
        api_pipeline.deploy_single_level(intervention, state_timeseries, summary_output, output)


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
def generate_top_counties(disable_validation, input_dir, output, state, fips):
    """The entry function for invocation"""
    intervention = Intervention.SELECTED_INTERVENTION
    active_states = [state.abbr for state in us.STATES] + ["PR"]
    us_latest = combined_datasets.load_us_latest_dataset().get_subset(
        AggregationLevel.COUNTY, states=active_states, state=state, fips=fips
    )
    us_timeseries = combined_datasets.load_us_timeseries_dataset().get_subset(
        AggregationLevel.COUNTY, states=active_states, state=state, fips=fips
    )

    def sort_func(output: RegionSummaryWithTimeseries):
        return -output.projections.totalHospitalBeds.peakShortfall

    all_timeseries = api_pipeline.run_on_all_fips_for_intervention(
        us_latest,
        us_timeseries,
        Intervention.SELECTED_INTERVENTION,
        input_dir,
        sort_func=sort_func,
        limit=100,
    )
    bulk_timeseries = AggregateRegionSummaryWithTimeseries(__root__=all_timeseries)

    api_pipeline.deploy_json_api_output(
        intervention, bulk_timeseries, output, filename_override="counties_top_100.json"
    )
    _logger.info("Finished top counties job")
