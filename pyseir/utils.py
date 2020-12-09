import os
from datetime import datetime
from enum import Enum

from scipy import signal

from libs.pipeline import Region

from pyseir import OUTPUT_DIR
from libs.datasets import combined_datasets
from libs.datasets.dataset_utils import AggregationLevel

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


class RunArtifact(Enum):
    RT_INFERENCE_REPORT = "rt_inference_report"
    RT_SMOOTHING_REPORT = "rt_smoothing_report"

    MLE_FIT_REPORT = "mle_fit_report"

    WEB_UI_RESULT = "web_ui_result"

    BACKTEST_RESULT = "backtest_result"


class SummaryArtifact(Enum):
    RT_METRIC_COMBINED = "rt_combined_metric.csv"
    ICU_METRIC_COMBINED = "icu_combined_metric.csv"


def get_summary_artifact_path(artifact: SummaryArtifact, output_dir=None) -> str:
    """
    Get an artifact path for a summary object

    Parameters
    ----------
    artifact: SummaryArtifact
        The artifact type to retrieve the pointer for.
    output_dir: str or NoneType
        Output directory to obtain the path for.

    Returns
    -------
    path: str
        Location of the artifact.
    """
    output_dir = output_dir or OUTPUT_DIR
    return os.path.join(output_dir, "pyseir", artifact.value)


def get_run_artifact_path(region: Region, artifact, output_dir=None) -> str:
    """
    Get an artifact path for a given locale and artifact type.

    Parameters
    ----------
    fips: str
        State or county fips code. Can also be a 2 character state abbreviation.
        If arbitrary string (e.g. for tests) then passed through
    artifact: RunArtifact
        The artifact type to retrieve the pointer for.
    output_dir: str or NoneType
        Output directory to obtain the path for.

    Returns
    -------
    path: str
        Location of the artifact.
    """

    if region.level is AggregationLevel.COUNTY:
        agg_level = AggregationLevel.COUNTY
        state_name = region.get_state_region().state_obj().name
        county = combined_datasets.get_county_name(region)
    elif region.level is AggregationLevel.STATE:
        agg_level = AggregationLevel.STATE
        state_name = region.state_obj().name
        county = None
    elif region.level is AggregationLevel.CBSA:
        agg_level = AggregationLevel.CBSA
        state_name = "CBSA"
        county = None
    else:
        raise AssertionError(f"Unsupported aggregation level {region.level}")

    fips = region.fips

    artifact = RunArtifact(artifact)

    output_dir = output_dir or OUTPUT_DIR

    if artifact is RunArtifact.RT_INFERENCE_REPORT:
        if agg_level is AggregationLevel.COUNTY:
            path = os.path.join(
                REPORTS_FOLDER(output_dir, state_name),
                f"Rt_results__{state_name}__{county}__{fips}.pdf",
            )
        else:
            path = os.path.join(
                STATE_SUMMARY_FOLDER(output_dir),
                "reports",
                f"Rt_results__{state_name}__{fips}.pdf",
            )

    elif artifact is RunArtifact.RT_SMOOTHING_REPORT:
        if agg_level is AggregationLevel.COUNTY:
            path = os.path.join(
                REPORTS_FOLDER(output_dir, state_name),
                f"Rt_smoothing__{state_name}__{county}__{fips}.pdf",
            )
        elif agg_level is AggregationLevel.STATE:
            path = os.path.join(
                STATE_SUMMARY_FOLDER(output_dir),
                "reports",
                f"Rt_smoothing__{state_name}__{fips}.pdf",
            )
        else:
            path = os.path.join(
                STATE_SUMMARY_FOLDER(output_dir),
                "reports",
                f"Rt_smoothing__{state_name}__{fips}.pdf",
            )

    elif artifact is RunArtifact.MLE_FIT_REPORT:
        if agg_level is AggregationLevel.COUNTY:
            path = os.path.join(
                REPORTS_FOLDER(output_dir, state_name),
                f"mle_fit_results__{state_name}__{county}__{fips}.pdf",
            )
        else:
            path = os.path.join(
                STATE_SUMMARY_FOLDER(output_dir),
                "reports",
                f"mle_fit_results__{state_name}__{fips}.pdf",
            )

    elif artifact is RunArtifact.WEB_UI_RESULT:
        path = os.path.join(WEB_UI_FOLDER(output_dir), f"{fips}.__INTERVENTION_IDX__.json")

    elif artifact is RunArtifact.BACKTEST_RESULT:
        if agg_level is AggregationLevel.COUNTY:
            path = os.path.join(
                REPORTS_FOLDER(output_dir, state_name),
                f"backtest_results__{state_name}__{county}__{fips}.pdf",
            )
        else:
            path = os.path.join(
                STATE_SUMMARY_FOLDER(output_dir),
                "reports",
                f"backtest_results__{state_name}__{fips}.pdf",
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
