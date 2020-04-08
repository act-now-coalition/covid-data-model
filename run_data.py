#!/usr/bin/env python
"""
CLI Functionality to generate reports and summaries of data sources.

To generate model output, see ./run_model.py
"""
import pathlib
import json
import logging
import click
from libs.datasets import JHUDataset
from libs.datasets import dataset_export
from libs.datasets import data_version
from libs.pipelines import latest_case_data_pipeline
_logger = logging.getLogger(__name__)


@click.group()
def main():
    pass


@main.command("latest")
@click.option(
    "--output",
    "-o",
    help="Output directory",
    type=pathlib.Path,
    default=pathlib.Path("results/case_summaries"),
)
@data_version.with_git_version_click_option
def run_latest(version: data_version.DataVersion, output: pathlib.Path):
    """Get latest case values from JHU dataset."""
    output.mkdir(exist_ok=True)
    timeseries = JHUDataset.local().timeseries()
    case_summary = latest_case_data_pipeline.build_summary(timeseries)
    state_summaries = latest_case_data_pipeline.build_output_for_api(case_summary)
    latest_case_data_pipeline.write_output(output, state_summaries, version)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
