import os
from enum import Enum

from scipy import signal

from libs.pipeline import Region

from pyseir import OUTPUT_DIR
from libs.datasets import combined_datasets
from libs.datasets.dataset_utils import AggregationLevel

REPORTS_FOLDER = lambda output_dir, state_name: os.path.join(
    output_dir, "pyseir", state_name, "reports"
)
STATE_SUMMARY_FOLDER = lambda output_dir: os.path.join(output_dir, "pyseir", "state_summaries")


class RunArtifact(Enum):
    RT_INFERENCE_REPORT = "rt_inference_report"
    RT_SMOOTHING_REPORT = "rt_smoothing_report"


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


def get_run_artifact_path(region: Region, artifact: RunArtifact, output_dir=None) -> str:
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

    output_dir = output_dir or OUTPUT_DIR

    if region.level is AggregationLevel.COUNTY:
        state_name = region.get_state_region().state_obj().name
        county = combined_datasets.get_county_name(region)
        readable_name = f"{state_name}__{county}__{region.fips}"
        folder = REPORTS_FOLDER(output_dir, state_name)
    elif region.level is AggregationLevel.STATE:
        state_name = region.state_obj().name
        readable_name = f"{state_name}__{region.fips}"
        folder = os.path.join(STATE_SUMMARY_FOLDER(output_dir), "reports")
    elif region.level is AggregationLevel.CBSA:
        readable_name = f"CBSA__{region.fips}"
        folder = os.path.join(STATE_SUMMARY_FOLDER(output_dir), "reports")
    elif region.level is AggregationLevel.PLACE:
        state_name = region.get_state_region().state_obj().name
        readable_name = f"{state_name}__{region.fips}"
        folder = os.path.join(STATE_SUMMARY_FOLDER(output_dir), "reports")
    elif region.level is AggregationLevel.COUNTRY:
        readable_name = region.country
        folder = os.path.join(output_dir, "pyseir", "reports")
    else:
        raise AssertionError(f"Unsupported aggregation level {region.level}")

    artifact = RunArtifact(artifact)

    if artifact is RunArtifact.RT_INFERENCE_REPORT:
        path = os.path.join(folder, f"Rt_results__{readable_name}.pdf")

    elif artifact is RunArtifact.RT_SMOOTHING_REPORT:
        path = os.path.join(folder, f"Rt_smoothing__{readable_name}.pdf")
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
