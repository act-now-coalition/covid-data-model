#!/usr/bin/env python

import pathlib
import json
import logging
import click
from libs import build_params
from libs.datasets import JHUDataset
from libs.datasets import FIPSPopulation
from libs.datasets.dataset_utils import AggregationLevel

WEB_DEPLOY_PATH = pathlib.Path("../covid-projections/public/data")


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
    output_dir = pathlib.Path(build_params.OUTPUT_DIR)
    if deploy:
        output_dir = WEB_DEPLOY_PATH

    output_folder = output_dir / "case_summary"
    output_folder.mkdir(exist_ok=True)
    timeseries = JHUDataset.local().timeseries().get_subset(None, country="USA")

    timeseries.data.cases = timeseries.data.cases.fillna(0)
    timeseries.data.deaths = timeseries.data.deaths.fillna(0)

    latest_state = timeseries.latest_values(AggregationLevel.STATE)
    latest_county = timeseries.latest_values(AggregationLevel.COUNTY)

    latest_state["date"] = latest_state["date"].dt.strftime("%Y-%m-%d")
    latest_county["date"] = latest_county["date"].dt.strftime("%Y-%m-%d")

    states = latest_state[["state", "cases", "deaths", "source", "date"]].to_dict(
        orient="records"
    )
    for state_data in states:
        state = state_data["state"]

        county_data = latest_county[latest_county.state == state]
        counties = county_data[["fips", "cases", "deaths", "source", "date"]].to_dict(
            orient="records"
        )

        state_data.update({"counties": counties})

        output_file = output_folder / f"{state}.summary.json"
        with output_file.open("w") as f:
            _logger.info(f"Writing latest data for {state}")
            json.dump(state_data, f)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARN)
    main()
