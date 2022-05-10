import dataclasses
import pathlib
from dataclasses import dataclass
from typing import List

import pandas as pd
import structlog
from datapublic.common_fields import CommonFields
from typing_extensions import final

from libs import pipeline
from libs.datasets.timeseries import MultiRegionDataset
from libs.datasets.timeseries import OneRegionTimeseriesDataset
from pyseir.rt import infer_rt
from pyseir.utils import SummaryArtifact

# TODO(tom): come up with cleaner handling of log object.
_log = structlog.get_logger()


@dataclass
class OneRegionPipeline:
    """Runs the pipeline for one region and stores the output."""

    region: pipeline.Region
    infer_df: pd.DataFrame
    _combined_data: OneRegionTimeseriesDataset

    @staticmethod
    def run(input: OneRegionTimeseriesDataset) -> "OneRegionPipeline":
        # `infer_df` does not have the NEW_ORLEANS patch applied. TODO(tom): Rename to something like
        # infection_rate.
        infer_rt_input = infer_rt.RegionalInput.from_regional_data(input)
        try:
            infer_df = infer_rt.run_rt(infer_rt_input)
        except Exception:
            _log.exception(f"run_rt failed for {input.region}")
            infer_df = pd.DataFrame()

        return OneRegionPipeline(region=input.region, infer_df=infer_df, _combined_data=input,)

    def population(self) -> float:
        return self._combined_data.latest[CommonFields.POPULATION]


@final
@dataclasses.dataclass(frozen=True)
class PyseirOutputDatasets:
    """Stores pyseir output from multiple regions in MultiRegionDataset objects."""

    infection_rate: MultiRegionDataset

    def write(self, output_dir: pathlib.Path, log):
        output_dir_path = pathlib.Path(output_dir)
        if not output_dir_path.exists():
            output_dir_path.mkdir()

        output_path = output_dir_path / SummaryArtifact.RT_METRIC_COMBINED.value
        self.infection_rate.to_csv(output_path)
        log.info(f"Saving Rt results to {output_path}")

    from libs.timing_utils import timing

    @staticmethod
    @timing
    def read(output_dir: pathlib.Path) -> "PyseirOutputDatasets":
        rt_data_path = output_dir / SummaryArtifact.RT_METRIC_COMBINED.value
        rt_data = MultiRegionDataset.from_csv(rt_data_path)

        return PyseirOutputDatasets(infection_rate=rt_data)

    @staticmethod
    def from_pipeline_output(pipelines: List[OneRegionPipeline]) -> "PyseirOutputDatasets":
        infection_rate_metric_df = pd.concat((p.infer_df for p in pipelines), ignore_index=True)
        infection_rate_ds = MultiRegionDataset.from_timeseries_df(infection_rate_metric_df)

        return PyseirOutputDatasets(infection_rate=infection_rate_ds)
