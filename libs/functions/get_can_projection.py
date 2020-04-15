import os
import json
import requests

from libs.enums import Intervention
from libs.datasets.dataset_utils import AggregationLevel
from libs.datasets.can_model_output_schema import CAN_MODEL_OUTPUT_SCHEMA


def _get_interventions_for_state(state):
    # TODO: read this from a dataset class
    interventions_url = "https://raw.githubusercontent.com/covid-projections/covid-projections/master/src/assets/data/interventions.json"
    interventions = requests.get(interventions_url).json()
    return interventions[state]


def _get_intervention(intervention, state):
    if intervention == Intervention.CURRENT.value:
        state_intervention = _get_interventions_for_state(state)
        return Intervention.from_str(state_intervention).value
    return intervention


def get_can_projection_path(
    input_dir, state_abbrev, fips, aggregation_level, initial_intervention
):
    intervention = _get_intervention(initial_intervention, state_abbrev)
    if aggregation_level == AggregationLevel.STATE:
        file_name = f"{state_abbrev}.{intervention.value}.json"
        data_directory = "state"
    else:
        file_name = f"{state_abbrev}.{fips}.{intervention.value}.json"
        data_directory = "county"
    file_path = os.path.join(input_dir, data_directory, file_name)
    return file_path


def standardize_json_data(json_data, schema_names):
    data_with_fields = []
    for row in json_data:
        data_row_with_fields = {}
        for i, field in enumerate(row):
            data_row_with_fields[schema_names[i]] = field
        data_with_fields.append(data_row_with_fields)
    return data_with_fields


def get_can_raw_data(input_dir, state_abbrev, fips, aggregation_level, intervention):
    file_path = get_can_projection_path(
        input_dir, state_abbrev, fips, aggregation_level, intervention
    )
    if os.path.exists(file_path): 
        with open(file_path) as json_file:
            return standardize_json_data(json.load(json_file), CAN_MODEL_OUTPUT_SCHEMA)
    # TODO : probably error out or log something here
    return []