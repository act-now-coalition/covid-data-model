import click
import requests

from qa.comparitor import Comparitor

@click.command("compare-snapshots")
@click.option(
    "--snapshot1",
    '-s1',
    type=int,
    help="The number of the snapshot to compare",
    required=True,
)
@click.option(
    "--snapshot2",
    '-s2',
    type=int,
    help="The number of the snapshot to compare with",
    required=True,
)
@click.option(
    "--intervention",
    type=str,
    help="Optional param that can specificy a specific intervention",
    required=False,
)
def compare_snapshots(snapshot1, snapshot2, intervention=None):
    if not intervention:
        # wow ugh i hate that this lives in a string....
        intervention = "SELECTED_INTERVENTION"
        
    states_comparitor = Comparitor(snapshot1, snapshot2, "CA", intervention)
    results = states_comparitor.compareMetrics()

    # complciations 1) the dates are off from the projects so need to match ond atse
    # each field maybe needs it's own thing it's own threshold it cares about
    # each this output is somewhat useless
    print(results)
