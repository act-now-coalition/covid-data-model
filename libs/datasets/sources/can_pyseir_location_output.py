import pathlib
import datetime

import pandas as pd


def get_can_projection_path(input_dir, fips, intervention) -> pathlib.Path:
    file_name = f"{fips}.{intervention.value}.json"
    file_path = pathlib.Path(input_dir) / file_name
    return file_path


class CANPyseirLocationOutput(object):
    def __init__(self, fips, data, intervention):
        self.fips = fips
        self.data = data
        self.intervention = intervention

    @classmethod
    def load_projection(cls, fips, intervention, input_dir):
        path = get_can_projection_path(input_dir, fips, intervention)
        data = pd.read_json(path)
        return cls(fips, data, intervention)

    @property
    def peak_hospitalizations_date(self) -> datetime.datetime:
        return self.data.iloc[self.data.all_hospitalized.idxmax()].date
