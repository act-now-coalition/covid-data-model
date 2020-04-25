import os
import json
import pandas as pd
from pyseir import cli
from pyseir.utils import get_run_artifact_path, RunArtifact
import libs.datasets.can_model_output_schema as schema


def test__pyseir_end_to_end():
    # This covers a lot of edge cases.
    cli._run_all(state='idaho')
    path = get_run_artifact_path('16001', RunArtifact.WEB_UI_RESULT).replace('__INTERVENTION_IDX__', '2')
    assert os.path.exists(path)

    with open(path) as f:
        output = json.load(f)

    assert (pd.DataFrame(output)[schema.CAN_MODEL_OUTPUT_SCHEMA.index(schema.RT_INDICATOR)].astype(float) > 0).any()
    assert (pd.DataFrame(output)[schema.CAN_MODEL_OUTPUT_SCHEMA.index(schema.RT_INDICATOR)].astype(float) > 6).any() is False
