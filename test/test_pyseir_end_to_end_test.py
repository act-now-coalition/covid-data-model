import os
import json
import numpy as np
from pyseir.utils import get_run_artifact_path, RunArtifact
from pyseir.cli import _run_all
import libs.datasets.can_model_output_schema as schema



def test__pyseir_end_to_end():
    # This covers a lot of edge cases.
    #_run_all(state='montana')
    path = get_run_artifact_path('29189', RunArtifact.WEB_UI_RESULT).replace('__INTERVENTION_IDX__', '2')
    assert os.path.exists(path)

    with open(path) as f:
        output = json.load(f)

    rt_series = np.array(output[schema.CAN_MODEL_OUTPUT_SCHEMA.index(schema.RT_INDICATOR)])
    assert (rt_series > 0).sum() > 1
    assert (rt_series > 5).sum() == 0
