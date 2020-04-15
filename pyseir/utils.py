import os
import us
from enum import Enum
from pyseir import OUTPUT_DIR
from pyseir import load_data
from libs.datasets.dataset_utils import AggregationLevel

REPORTS_FOLDER = lambda output_dir, state_name: os.path.join(output_dir, 'pyseir', state_name, 'reports')
DATA_FOLDER = lambda output_dir, state_name: os.path.join(output_dir, 'pyseir', state_name, 'data')
WEB_UI_FOLDER = lambda output_dir: os.path.join(output_dir, 'web_ui')
STATE_SUMMARY_FOLDER = lambda output_dir: os.path.join(output_dir, 'pyseir', 'state_summaries')


class RunMode(Enum):
    # Read params from the parameter sampler default and use empirical
    # suppression policies.
    DEFAULT = 'default'
    # 4 basic suppression scenarios and specialized parameters to match
    # covidactnow before scenarios.  Uses hospitalization data to fix.
    CAN_BEFORE_HOSPITALIZATION = 'can-before-hospitalization'
    # Same as CAN Before but with updated ICU, hosp rates increased.
    CAN_BEFORE_HOSPITALIZATION_NEW_PARAMS = 'can-before-hospitalization-new-params'


class RunArtifact(Enum):
    MLE_FIT_RESULT = 'mle_fit_result'
    MLE_FIT_MODEL = 'mle_fit_model'
    MLE_FIT_REPORT = 'mle_fit_report'

    ENSEMBLE_RESULT = 'ensemble_result'
    ENSEMBLE_REPORT = 'ensemble_report'

    WEB_UI_RESULT = 'web_ui_result'


def get_run_artifact_path(fips, artifact, output_dir=None):
    """
    Get an artifact path for a given locale and artifact type.

    Parameters
    ----------
    fips: str
        State or county fips code. Can also be a 2 character state abbreviation.
    artifact: RunArtifact
        The artifact type to retrieve the pointer for.
    output_dir: str or NoneType
        Output directory to obtain the path for.

    Returns
    -------
    path: str
        Location of the artifact.
    """
    state_obj = us.states.lookup(fips[:2])
    if len(fips) == 5:
        agg_level = AggregationLevel.COUNTY
        county = load_data.load_county_metadata_by_fips(fips)['county']
    else:
        agg_level = AggregationLevel.STATE

    artifact = RunArtifact(artifact)

    output_dir = output_dir or OUTPUT_DIR

    if artifact is RunArtifact.MLE_FIT_REPORT:
        if agg_level is AggregationLevel.COUNTY:
            path = os.path.join(REPORTS_FOLDER(output_dir, state_obj.name), f'mle_fit_results__{state_obj.name}__{county}__{fips}.pdf')
        else:
            path = os.path.join(STATE_SUMMARY_FOLDER(output_dir), 'reports', f'mle_fit_results__{state_obj.name}__{fips}.pdf')

    elif artifact is RunArtifact.MLE_FIT_RESULT:
        if agg_level is AggregationLevel.COUNTY:
            path = os.path.join(STATE_SUMMARY_FOLDER(output_dir), 'data', f'mle_fit_results__{state_obj.name}_counties.json')
        else:
            path = os.path.join(STATE_SUMMARY_FOLDER(output_dir), 'data', f'mle_fit_results__{state_obj.name}_state_only.json')

    elif artifact is RunArtifact.MLE_FIT_MODEL:
        if agg_level is AggregationLevel.COUNTY:
            path = os.path.join(DATA_FOLDER(output_dir, state_obj.name), f'mle_fit_model__{state_obj.name}__{county}__{fips}.pkl')
        else:
            path = os.path.join(STATE_SUMMARY_FOLDER(output_dir), 'data', f'mle_fit_model__{state_obj.name}_state_only.pkl')

    elif artifact is RunArtifact.ENSEMBLE_RESULT:
        if agg_level is AggregationLevel.COUNTY:
            path = os.path.join(DATA_FOLDER(output_dir, state_obj.name), f'ensemble_projections__{state_obj.name}__{county}__{fips}.json')
        else:
            path = os.path.join(STATE_SUMMARY_FOLDER(output_dir), 'data', f'ensemble_projections__{state_obj.name}__{fips}.json')

    elif artifact is RunArtifact.ENSEMBLE_REPORT:
        if agg_level is AggregationLevel.COUNTY:
            path = os.path.join(REPORTS_FOLDER(output_dir, state_obj.name), f'ensemble_projections__{state_obj.name}__{county}__{fips}.pdf')
        else:
            path = os.path.join(STATE_SUMMARY_FOLDER(output_dir), 'reports', f'ensemble_projections__{state_obj.name}__{fips}.pdf')

    elif artifact is RunArtifact.WEB_UI_RESULT:
        if agg_level is AggregationLevel.COUNTY:

            path = os.path.join(WEB_UI_FOLDER(output_dir), 'county', f'{state_obj.abbr}.{fips}.__INTERVENTION_IDX__.json')
        else:
            path = os.path.join(WEB_UI_FOLDER(output_dir), 'state', f'{state_obj.abbr}.__INTERVENTION_IDX__.json')

    else:
        raise ValueError(f'No paths available for artifact {RunArtifact}')

    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path
