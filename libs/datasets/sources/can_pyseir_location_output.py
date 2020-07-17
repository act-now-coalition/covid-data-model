from typing import Optional
import pathlib
import datetime

import pandas as pd

from libs.datasets import can_model_output_schema as schema
from libs.enums import Intervention
from pyseir.deployment import webui_data_adaptor_v1


def get_can_projection_path(input_dir, fips, intervention) -> pathlib.Path:
    file_name = f"{fips}.{intervention.value}.json"
    file_path = pathlib.Path(input_dir) / file_name
    return file_path


def _calculate_shortfall(beds: pd.Series, hospitalized: pd.Series) -> pd.Series:
    shortfall = hospitalized - beds
    shortfall[shortfall > 0] = 0
    return shortfall.abs()


class CANPyseirLocationOutput(object):
    def __init__(self, data):
        data = data.reset_index(drop=True)

        self.fips = data[schema.FIPS].iloc[0]
        self.data = data
        self.intervention = Intervention(data[schema.INTERVENTION].iloc[0])

        self.data["short_fall"] = _calculate_shortfall(
            self.data[schema.ALL_HOSPITALIZED], self.data[schema.BEDS]
        )

    @classmethod
    def load_from_path(cls, path):
        data = pd.read_json(
            path,
            convert_dates=[schema.DATE],
            dtype={schema.FIPS: str},
            orient=webui_data_adaptor_v1.OUTPUT_JSON_ORIENT,
        )
        return cls(data)

    @classmethod
    def load_from_model_output_if_exists(
        cls, fips, intervention, input_dir
    ) -> Optional["CANPyseirLocationOutput"]:
        path = get_can_projection_path(input_dir, fips, intervention)
        if not path.exists():
            return None

        return cls.load_from_path(path)

    @property
    def peak_hospitalizations_date(self) -> datetime.datetime:
        return self.data.iloc[self.data.all_hospitalized.idxmax()].date.to_pydatetime()

    @property
    def hospitals_shortfall_date(self) -> Optional[datetime.datetime]:
        if not self.peak_hospitalizations_shortfall:
            return None

        shortfall = self.data["short_fall"]
        return self.data[shortfall > 0].iloc[0].date.to_pydatetime()

    @property
    def peak_hospitalizations_shortfall(self):
        # Need to predict this.
        return self.data["short_fall"].max()

    @property
    def latest_rt(self) -> Optional[float]:
        """Returns the latest Rt value from Rt inference process."""
        last_idx = self.data[schema.RT_INDICATOR].last_valid_index()
        if last_idx is None:
            return None

        return self.data.loc[last_idx][schema.RT_INDICATOR]

    @property
    def latest_rt_ci90(self) -> Optional[float]:
        """Returns the latest Rt confidence bounds from Rt inference process."""
        last_idx = self.data[schema.RT_INDICATOR_CI90].last_valid_index()
        if last_idx is None:
            return None

        return self.data.loc[last_idx][schema.RT_INDICATOR_CI90]
