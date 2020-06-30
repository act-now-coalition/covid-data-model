from typing import Optional
import pathlib
import datetime

import pandas as pd

from libs.datasets import can_model_output_schema as schema
from libs.enums import Intervention


def get_can_projection_path(input_dir, fips, intervention) -> pathlib.Path:
    file_name = f"{fips}.{intervention.value}.json"
    file_path = pathlib.Path(input_dir) / file_name
    return file_path


class CANPyseirLocationOutput(object):
    def __init__(self, data):
        self.fips = data[schema.FIPS].iloc[0]
        self.data = data
        self.intervention = Intervention(data[schema.INTERVENTION].iloc[0])

        self.data["short_fall"] = self.data[schema.BEDS] - self.data[schema.ALL_HOSPITALIZED]

    @classmethod
    def load_from_path(cls, path):
        data = pd.read_json(path, convert_dates=[schema.DATE], dtype={schema.FIPS: str})
        return cls(data)

    @classmethod
    def load_from_model_output_if_exists(
        cls, fips, intervention, input_dir
    ) -> Optional["CANPyseirLocationOutput"]:
        path = get_can_projection_path(input_dir, fips, intervention)
        print(path)
        if not path.exists():
            print(intervention)
            return None

        return cls.load_from_path(path)

    @property
    def peak_hospitalizations_date(self) -> datetime.datetime:
        return self.data.iloc[self.data.all_hospitalized.idxmax()].date.to_pydatetime()

    @property
    def hospitals_shortfall_date(self) -> Optional[datetime.datetime]:
        is_short_fall = self.data["short_fall"] < 0

        if not sum(is_short_fall):
            return None
        return self.data.loc[is_short_fall, schema.DATE].iloc[0]

    @property
    def peak_hospitalizations_shortfall(self):
        # Need to predict this.
        return self.data.iloc[self.data[schema.ALL_HOSPITALIZED].idxmax()].short_fall or 0

    @property
    def latest_rt(self) -> float:
        return self.data.iloc[-1][schema.Rt]

    @property
    def latest_rt_ci90(self) -> float:
        return self.data.iloc[-1][schema.Rt_ci90]
