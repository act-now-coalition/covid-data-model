import os
import us
from datetime import datetime
from enum import Enum
from scipy import signal
from pyseir import OUTPUT_DIR
from pyseir import load_data
from libs.datasets.dataset_utils import AggregationLevel

import structlog
from structlog.threadlocal import bind_threadlocal, clear_threadlocal, merge_threadlocal
from structlog import configure

configure(processors=[merge_threadlocal, structlog.processors.KeyValueRenderer()])
log = structlog.get_logger(__name__)

REPORTS_FOLDER = lambda output_dir, state_name: os.path.join(
    output_dir, "pyseir", state_name, "reports"
)
DATA_FOLDER = lambda output_dir, state_name: os.path.join(output_dir, "pyseir", state_name, "data")
WEB_UI_FOLDER = lambda output_dir: os.path.join(output_dir, "web_ui")
STATE_SUMMARY_FOLDER = lambda output_dir: os.path.join(output_dir, "pyseir", "state_summaries")
REF_DATE = datetime(year=2020, month=1, day=1)


class TimeseriesType(Enum):
    RAW_NEW_CASES = "raw_new_cases"
    RAW_NEW_DEATHS = "raw_new_deaths"
    NEW_CASES = "new_cases"
    NEW_DEATHS = "new_deaths"
    NEW_HOSPITALIZATIONS = "new_hospitalizations"
    CURRENT_HOSPITALIZATIONS = "current_hospitalizations"
    NEW_TESTS = "new_tests"


class RunMode(Enum):
    DEFAULT = "default"

    # Inference based + future suppression policy.
    CAN_INFERENCE_DERIVED = "can-inference-derived"


class RunArtifact(Enum):

    FORECAST_VAR_UNSCALED = "forecast_var_unscaled"
    FORECAST_VAR_SCALED = "forecast_var_scaled"
    FORECAST_RESULT = "forecast_result"
    FORECAST_LOSS = "forecast_loss"

    RT_SMOOTHING_REPORT = "rt_smoothing_report"
    RT_INFERENCE_RESULT = "rt_inference_result"
    RT_INFERENCE_REPORT = "rt_inference_report"

    MLE_FIT_RESULT = "mle_fit_result"
    MLE_FIT_MODEL = "mle_fit_model"
    MLE_FIT_REPORT = "mle_fit_report"

    WHITELIST_RESULT = "whitelist_result"

    ENSEMBLE_RESULT = "ensemble_result"
    ENSEMBLE_REPORT = "ensemble_report"

    WEB_UI_RESULT = "web_ui_result"

    BACKTEST_RESULT = "backtest_result"


def get_run_artifact_path(fips, artifact, output_dir=None) -> str:
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
    log.info(f"getting state obj fips: {fips}")
    state_obj = us.states.lookup("01")  # [:2])
    log.info(state_obj)

    if len(fips) == 5:
        agg_level = AggregationLevel.COUNTY
        county = load_data.load_county_metadata_by_fips(fips)["county"]
    else:
        agg_level = AggregationLevel.STATE

    log.info("getting artifact")
    artifact = RunArtifact(artifact)
    log.info("got artifact")
    log.info(artifact)

    output_dir = output_dir or OUTPUT_DIR

    if artifact is RunArtifact.RT_INFERENCE_REPORT:
        if agg_level is AggregationLevel.COUNTY:
            path = os.path.join(
                REPORTS_FOLDER(output_dir, state_obj.name),
                f"Rt_results__{state_obj.name}__{county}__{fips}.pdf",
            )
        else:
            path = os.path.join(
                STATE_SUMMARY_FOLDER(output_dir),
                "reports",
                f"Rt_results__{state_obj.name}__{fips}.pdf",
            )

    elif artifact is RunArtifact.RT_SMOOTHING_REPORT:
        if agg_level is AggregationLevel.COUNTY:
            path = os.path.join(
                REPORTS_FOLDER(output_dir, state_obj.name),
                f"Rt_smoothing__{state_obj.name}__{county}__{fips}.pdf",
            )
        else:
            path = os.path.join(
                STATE_SUMMARY_FOLDER(output_dir),
                "reports",
                f"Rt_smoothing__{state_obj.name}__{fips}.pdf",
            )

    elif artifact is RunArtifact.RT_INFERENCE_RESULT:
        if agg_level is AggregationLevel.COUNTY:
            path = os.path.join(
                DATA_FOLDER(output_dir, state_obj.name),
                f"Rt_results__{state_obj.name}__{county}__{fips}.json",
            )
        else:
            path = os.path.join(
                STATE_SUMMARY_FOLDER(output_dir),
                "data",
                f"Rt_results__{state_obj.name}__{fips}.json",
            )

    elif artifact is RunArtifact.MLE_FIT_REPORT:
        if agg_level is AggregationLevel.COUNTY:
            path = os.path.join(
                REPORTS_FOLDER(output_dir, state_obj.name),
                f"mle_fit_results__{state_obj.name}__{county}__{fips}.pdf",
            )
        else:
            path = os.path.join(
                STATE_SUMMARY_FOLDER(output_dir),
                "reports",
                f"mle_fit_results__{state_obj.name}__{fips}.pdf",
            )

    elif artifact is RunArtifact.MLE_FIT_RESULT:
        if agg_level is AggregationLevel.COUNTY:
            path = os.path.join(
                STATE_SUMMARY_FOLDER(output_dir),
                "data",
                f"mle_fit_results__{state_obj.name}_counties.json",
            )
        else:
            path = os.path.join(
                STATE_SUMMARY_FOLDER(output_dir),
                "data",
                f"mle_fit_results__{state_obj.name}_state_only.json",
            )

    elif artifact is RunArtifact.MLE_FIT_MODEL:
        if agg_level is AggregationLevel.COUNTY:
            path = os.path.join(
                DATA_FOLDER(output_dir, state_obj.name),
                f"mle_fit_model__{state_obj.name}__{county}__{fips}.pkl",
            )
        else:
            path = os.path.join(
                STATE_SUMMARY_FOLDER(output_dir),
                "data",
                f"mle_fit_model__{state_obj.name}_state_only.pkl",
            )

    elif artifact is RunArtifact.ENSEMBLE_RESULT:
        if agg_level is AggregationLevel.COUNTY:
            path = os.path.join(
                DATA_FOLDER(output_dir, state_obj.name),
                f"ensemble_projections__{state_obj.name}__{county}__{fips}.json",
            )
        else:
            path = os.path.join(
                STATE_SUMMARY_FOLDER(output_dir),
                "data",
                f"ensemble_projections__{state_obj.name}__{fips}.json",
            )

    elif artifact is RunArtifact.ENSEMBLE_REPORT:
        if agg_level is AggregationLevel.COUNTY:
            path = os.path.join(
                REPORTS_FOLDER(output_dir, state_obj.name),
                f"ensemble_projections__{state_obj.name}__{county}__{fips}.pdf",
            )
        else:
            path = os.path.join(
                STATE_SUMMARY_FOLDER(output_dir),
                "reports",
                f"ensemble_projections__{state_obj.name}__{fips}.pdf",
            )

    elif artifact is RunArtifact.WEB_UI_RESULT:
        path = os.path.join(WEB_UI_FOLDER(output_dir), f"{fips}.__INTERVENTION_IDX__.json",)

    elif artifact is RunArtifact.WHITELIST_RESULT:
        path = os.path.join(output_dir, "api_whitelist.json")

    elif artifact is RunArtifact.BACKTEST_RESULT:
        if agg_level is AggregationLevel.COUNTY:
            path = os.path.join(
                REPORTS_FOLDER(output_dir, state_obj.name),
                f"backtest_results__{state_obj.name}__{county}__{fips}.pdf",
            )
        else:
            path = os.path.join(
                STATE_SUMMARY_FOLDER(output_dir),
                "reports",
                f"backtest_results__{state_obj.name}__{fips}.pdf",
            )

    elif artifact is RunArtifact.FORECAST_VAR_UNSCALED:
        path = os.path.join(
            STATE_SUMMARY_FOLDER(output_dir),
            "reports",
            f"Forecast_var_unscaled_{state_obj.name}__{fips}.pdf",
        )

    elif artifact is RunArtifact.FORECAST_LOSS:
        path = os.path.join(STATE_SUMMARY_FOLDER(output_dir), "reports", f"Forecast_loss.pdf",)

    elif artifact is RunArtifact.FORECAST_VAR_SCALED:
        path = os.path.join(
            STATE_SUMMARY_FOLDER(output_dir),
            "reports",
            f"Forecast_var_scaled_{state_obj.name}__{fips}.pdf",
        )

    elif artifact is RunArtifact.FORECAST_RESULT:
        path = os.path.join(
            STATE_SUMMARY_FOLDER(output_dir),
            "reports",
            f"Forecast_result_{state_obj.name}__{fips}.pdf",
        )

    else:
        raise ValueError(f"No paths available for artifact {RunArtifact}")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def ewma_smoothing(series, tau=5):
    """
    Exponentially weighted moving average of a series.

    Parameters
    ----------
    series: array-like
        Series to convolve.
    tau: float
        Decay factor.

    Returns
    -------
    smoothed: array-like
        Smoothed series.
    """
    exp_window = signal.exponential(2 * tau, 0, tau, False)[::-1]
    exp_window /= exp_window.sum()
    smoothed = signal.convolve(series, exp_window, mode="same")
    return smoothed
