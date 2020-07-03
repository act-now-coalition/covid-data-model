import api
import logging
import pathlib
import click
import itertools
import us
from api.can_api_definition import CovidActNowAreaTimeseries
from api.can_api_definition import CovidActNowBulkTimeseries
from libs.pipelines import api_pipeline
from libs.datasets.dataset_utils import AggregationLevel
from libs.datasets import combined_datasets
from libs.enums import Intervention
from libs.datasets.dataset_utils import AggregationLevel

PROD_BUCKET = "data.covidactnow.org"

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
def update_schemas(output_dir):
    """Updates all public facing API schemas."""
    schemas = api.find_public_model_classes()
    for schema in schemas:
        path = output_dir / f"{schema.__name__}.json"
        _logger.info(f"Updating schema {schema} to {path}")
        path.write_text(schema.schema_json(indent=2))


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
    us_latest = combined_datasets.build_us_latest_with_all_fields().get_subset(
        aggregation_level, state=state, fips=fips, states=active_states
    )
    us_timeseries = combined_datasets.build_us_timeseries_with_all_fields().get_subset(
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
    "--disable-validation", "-dv", is_flag=True, help="Run the validation on the deploy command",
)
@click.option(
    "--input-dir", "-i", default="results", help="Input directory of state/county projections",
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
    active_states = [state.abbr for state in us.STATES]
    us_latest = combined_datasets.build_us_latest_with_all_fields().get_subset(
        AggregationLevel.COUNTY, states=active_states, state=state, fips=fips
    )
    us_timeseries = combined_datasets.build_us_timeseries_with_all_fields().get_subset(
        AggregationLevel.COUNTY, states=active_states, state=state, fips=fips
    )

    def sort_func(output: CovidActNowAreaTimeseries):
        return -output.projections.totalHospitalBeds.peakShortfall

    all_timeseries = api_pipeline.run_on_all_fips_for_intervention(
        us_latest,
        us_timeseries,
        Intervention.SELECTED_INTERVENTION,
        input_dir,
        sort_func=sort_func,
        limit=100,
    )
    bulk_timeseries = CovidActNowBulkTimeseries(__root__=all_timeseries)

    api_pipeline.deploy_json_api_output(
        intervention, bulk_timeseries, output, filename_override="counties_top_100.json"
    )
    # top_counties_pipeline.deploy_results(county_results_api, "counties_top_100", output)

    # _logger.info("finished top counties job")
