import click
import requests

from libs.qa.comparitor import Comparitor
from libs.enums import Intervention
from libs import dataset_deployer
from libs.us_state_abbrev import US_STATE_ABBREV


def _get_state_data(snapshot1, snapshot2, state_abbrev, intervention):
    # counties_url_1 = f'https://data.covidactnow.org/snapshot/{snapshot1}/us/counties.{intervention}.timeseries.json'
    # counties_url_2= f'https://data.covidactnow.org/snapshot/{snapshot2}/us/counties.{intervention}.timeseries.json'

    states_url_1 = f"https://data.covidactnow.org/snapshot/{snapshot1}/us/states/{state_abbrev}.{intervention}.timeseries.json"
    states_url_2 = f"https://data.covidactnow.org/snapshot/{snapshot2}/us/states/{state_abbrev}.{intervention}.timeseries.json"

    # counties_json_1 = requests.get(counties_url_1).json()
    # counties_json_2 = requests.get(counties_url_2).json()
    try:
        states_json_1 = requests.get(states_url_1).json()
        states_json_2 = requests.get(states_url_2).json()
    except:
        states_json_1, states_json_2 = None, None

    return states_json_1, states_json_2


@click.command("compare-snapshots")
@click.option(
    "--snapshot1", "-s1", type=int, help="The number of the snapshot to compare", required=True,
)
@click.option(
    "--snapshot2",
    "-s2",
    type=int,
    help="The number of the snapshot to compare with",
    required=True,
)
@click.option(
    "--state", type=str, help="Optional param that can specificy a specific state", required=False,
)
@click.option(
    "--fips", type=str, help="Optional param that can specificy a specific state", required=False,
)
@click.option(
    "--intervention",
    type=Intervention,
    help="Optional param that can specificy a specific intervention",
    required=False,
)
def compare_snapshots(snapshot1, snapshot2, state, fips, intervention):
    states = [state] if state else US_STATE_ABBREV.values()
    results = []

    for state_abbrev in states:
        if not intervention:
            intervention = Intervention.SELECTED_INTERVENTION
        if not fips:
            api1, api2 = _get_state_data(snapshot1, snapshot2, state_abbrev, intervention.name)
        else:
            raise NotImplementedError("currently only handles states data")
        if not (api1 and api2):
            print(f"State Abbrev {state_abbrev} doesn't have data")
            continue
        comparitor = Comparitor(
            snapshot1, snapshot2, api1, api2, state_abbrev, intervention.name, fips
        )
        state_results = comparitor.compareMetrics()
        results.extend(state_results)

    dataset_deployer.write_nested_csv(
        Comparitor.dict_results(sorted(results, reverse=True)), "compared", "output/"
    )
