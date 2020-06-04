import click
import requests
import os
import json
import re

import pathlib

from libs.qa.comparitor import Comparitor
from libs.enums import Intervention
from libs import dataset_deployer
from libs.us_state_abbrev import STATES_50


def _get_state_data(snapshot, state_abbrev, intervention):
    states_url_1 = f"https://data.covidactnow.org/snapshot/{snapshot}/us/states/{state_abbrev}.{intervention}.timeseries.json"

    try:
        states_json_1 = requests.get(states_url_1).json()
    except:
        states_json_1 = None

    return states_json_1


def _get_compare_snapshot():
    data_url_json = requests.get(
        "https://raw.githubusercontent.com/covid-projections/covid-projections/develop/src/assets/data/data_url.json"
    ).json()
    data_url = data_url_json["data_url"]
    match = re.findall(r"https://data.covidactnow.org/snapshot/(\d+)/", data_url)
    if not match:
        raise Exception("could not find data_url")
    return match[0]


def _get_input_dir_snapshot(input_dir):
    match = re.findall(r"/data/api-results-(\d+)/\w+", input_dir)
    if not match:
        return None
    return match[0]


@click.command("compare-snapshots")
@click.option(
    "--input-dir", "-i", type=str, help="The input path of the snapshot to compare", required=False,
)
@click.option(
    "--output-dir", "-o", type=str, help="The output directory to place", required=False,
)
@click.option(
    "--input-snapshot",
    "-s1",
    type=str,
    help="The number of the snapshot to compare",
    required=False,
)
@click.option(
    "--compare_snapshot",
    "-s2",
    type=int,
    help="The number of the snapshot to compare with. If not provided uses the latest on covidactnow develop branch",
    required=False,
)
@click.option(
    "--state", type=str, help="Optional param that can specificy a specific state", required=False,
)
@click.option(
    "--fips", type=str, help="Optional param that can specificy a specific state", required=False,
)
@click.option(
    "--intervention_name",
    type=str,
    default=Intervention.OBSERVED_INTERVENTION.name,
    help="Optional param that can specificy a specific intervention",
    required=False,
)
def compare_snapshots(
    input_dir, output_dir, input_snapshot, compare_snapshot, state, fips, intervention_name
):
    if not (input_dir or input_snapshot) or (input_dir and input_snapshot):
        raise Exception("Need to specify either snapshot or input dir not both")
    states = [state] if state else STATES_50.values()
    results = []
    report = []
    output_report = ""

    if not (compare_snapshot):
        compare_snapshot = _get_compare_snapshot()
        if input_dir:
            input_snapshot_from_dir = _get_input_dir_snapshot(input_dir)
            output_report += f"More info can be found at https://data.covidactnow.org/snapshot/{input_snapshot_from_dir}/qa/compared.csv"

    for state_abbrev in states:
        if not fips:
            if not input_dir:
                api1 = _get_state_data(input_snapshot, state_abbrev, intervention_name)
            else:
                filepath = os.path.join(
                    input_dir, f"{state_abbrev.upper()}.{intervention_name}.timeseries.json"
                )
                with open(filepath) as json_file:
                    api1 = json.load(json_file)
            api2 = _get_state_data(compare_snapshot, state_abbrev, intervention_name)
        else:
            raise NotImplementedError("currently only handles states data")
        if not (api1 and api2):
            print(f"State Abbrev {state_abbrev} doesn't have data")
        else:
            comparitor = Comparitor(
                input_snapshot, compare_snapshot, api1, api2, state_abbrev, intervention_name, fips
            )
            state_results = comparitor.compare_metrics()
            if state_results:
                report.append(comparitor.generate_report())
                results.extend(state_results)
            print(f"Adding {state_abbrev} {len(state_results)} to results")

    if not len(results):
        return f"Applied dif from {input_snapshot} to {compare_snapshot}, no difference above thresholds found"

    # make the output directory if it doesn't exist already
    os.makedirs(output_dir, exist_ok=True)
    dataset_deployer.write_nested_csv(
        Comparitor.dict_results(sorted(results, reverse=True)), "compared", output_dir
    )

    formatted_report = "\n--" + "\n\n--".join(report)
    output_report = f"Applied dif from {input_snapshot if input_snapshot else input_dir} to {compare_snapshot}: {formatted_report} {output_report}"
    # write report to a file
    output_path = pathlib.Path(output_dir) / f"report.txt"
    with output_path.open("w") as report_file:
        report_file.write(output_report)

    # send report to slack
    slack_url = os.getenv("SLACK_DEV_ALERTS_WEBHOOK")
    response = requests.post(
        slack_url,
        data=json.dumps({"text": output_report}),
        headers={"Content-Type": "application/json"},
    )
