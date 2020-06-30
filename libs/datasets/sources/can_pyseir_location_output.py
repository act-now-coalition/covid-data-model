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

    @classmethod
    def load_from_path(cls, path):
        data = pd.read_json(path, convert_dates=[schema.DATE], dtype={schema.FIPS: str})
        return cls(data)

    @classmethod
    def load_from_model_output(cls, fips, intervention, input_dir):
        path = get_can_projection_path(input_dir, fips, intervention)
        return cls.load_from_path(fips, intervention, path)

    @property
    def peak_hospitalizations_date(self) -> datetime.datetime:
        return self.data.iloc[self.data.all_hospitalized.idxmax()].date.to_pydatetime()
