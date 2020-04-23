import click
import logging
import os

from libs.pipelines import api_pipeline
from libs.datasets.dataset_utils import AggregationLevel
from libs.enums import Intervention

logger = logging.getLogger(__name__)
PROD_BUCKET = "data.covidactnow.org"


@click.command("deploy-counties-api")
@click.option(
    "--disable-validation",
    "-dv",
    is_flag=True,
    help="Run the validation on the deploy command",
)
@click.option(
    "--input-dir",
    "-i",
    default="results",
    help="Input directory of county projections",
)
@click.option(
    "--output",
    "-o",
    default="results/output/counties",
    help="Output directory for artifacts",
)
@click.option(
    "--summary-output",
    default="results/output",
    help="Output directory for county summaries.",
)
def deploy_counties_api(disable_validation, input_dir, output, summary_output):
    """The entry function for invocation"""

    # check that the dirs exist before starting
    for directory in [input_dir, output, summary_output]:
        if not os.path.isdir(directory):
            raise NotADirectoryError(directory)

    for intervention in list(Intervention):
        # TODO(issues/#258): remove check once counties support inferrence
        if intervention in Intervention.county_supported_interventions():
            county_result = api_pipeline.run_projections(
                input_dir,
                AggregationLevel.COUNTY,
                intervention,
                run_validation=not disable_validation,
            )
            county_summaries, county_timeseries = api_pipeline.generate_api(
                county_result, input_dir
            )
            api_pipeline.deploy_results([*county_summaries, *county_timeseries], output)

            counties_summary = api_pipeline.build_counties_summary(county_summaries, intervention)
            counties_timeseries = api_pipeline.build_counties_timeseries(county_timeseries, intervention)
            summarized_timeseries = api_pipeline.build_prediction_header_timeseries_data(counties_timeseries)
            api_pipeline.deploy_prediction_timeseries_csvs(summarized_timeseries, summary_output)

            api_pipeline.deploy_results([counties_summary], summary_output, write_csv=True)
            api_pipeline.deploy_results([counties_timeseries], summary_output)

        logger.info("finished top counties job")


@click.command("county-fips-summaries")
@click.option(
    "--input-dir",
    "-i",
    default="results",
    help="Input directory of county projections",
)
@click.option(
    "--output",
    "-o",
    default="results/county_summaries",
    help="Output directory for artifacts",
)
def county_fips_summaries(input_dir, output):
    """Generates sumary files by state and globally of counties with model output data."""
    county_summaries = api_pipeline.build_county_summary_from_model_output(input_dir)
    api_pipeline.deploy_results(county_summaries, output)
