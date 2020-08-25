import requests
from libs.enums import Intervention
import functools


@functools.lru_cache(None)
def get_interventions():
    interventions_url = "https://raw.githubusercontent.com/covid-projections/covid-data-public/master/data/misc/interventions.json"
    interventions = requests.get(interventions_url).json()
    return interventions


def get_intervention_for_state(state) -> Intervention:
    return Intervention.from_str(get_interventions()[state])
    # TODO: read this from a dataset class
    # interventions_url = "https://raw.githubusercontent.com/covid-projections/covid-data-public/master/data/misc/interventions.json"
    # interventions = requests.get(interventions_url).json()
    # return Intervention.from_str(interventions[state])


def _get_intervention(intervention, state):
    if intervention == Intervention.SELECTED_INTERVENTION:
        return get_intervention_for_state(state)
    return intervention
