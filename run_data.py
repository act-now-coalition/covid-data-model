#!/usr/bin/env python
"""
CLI Functionality to generate reports and summaries of data sources.

To generate model output, see ./run_model.py
"""
import json
import logging
import click
from libs import build_params
from libs.datasets import JHUDataset
from libs.datasets import dataset_export

_logger = logging.getLogger(__name__)


@click.group()
def main():
    pass


@main.command("latest")
@click.option(
    "--deploy",
    is_flag=True,
    help="Output data files to public data directory in local covid-projections.",
)
def run_latest(deploy=False):
    """Get latest case values from JHU dataset."""
    output_base = build_params.OUTPUT_PATH
    if deploy:
        output_base = build_params.WEB_DEPLOY_PATH

    output_folder = output_base / build_params.STATE_SUMMARY_SUBDIR
    output_folder.mkdir(exist_ok=True)
    timeseries = JHUDataset.local().timeseries()
    state_summaries = dataset_export.latest_case_summaries_by_state(timeseries)

    for state, state_summary in state_summaries:
        output_file = output_folder / f"{state}.summary.json"
        with output_file.open("w") as f:
            _logger.info(f"Writing latest data for {state}")
            json.dump(state_summary, f)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
