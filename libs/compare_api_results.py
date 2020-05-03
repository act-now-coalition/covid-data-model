import os
import json
import requests
from api.can_api_definition import CovidActNowStatesSummary
from api.can_api_definition import CovidActNowStateSummary
from api.can_api_definition import CovidActNowCountySummary
from api.can_api_definition import CovidActNowStateTimeseries
from api.can_api_definition import CovidActNowCountyTimeseries


CAN_API_BASE_URL = "https://data.covidactnow.org/"


class APISnapshot(object):

    def __init__(self, root: str):
        self.root = root
        self._is_url = self.root.startswith("https://")

    def _load_path(self, path):
        if self._is_url:
            return requests.get(path).json()

        with open(path) as f:
            return json.load(f)

    @classmethod
    def from_snapshot(cls, snapshot_id: int):
        path = os.path.join(CAN_API_BASE_URL,  f"snapshot/{snapshot_id}")
        return cls(path)

    @classmethod
    def from_latest(cls):
        path = os.path.join(CAN_API_BASE_URL,  f"latest")
        return cls(path)

    def state_summary(self, state, intervention) -> CovidActNowStateSummary:
        path = os.path.join(
            self.root, f"us/states/{state}.{intervention.name}.json"
        )
        data = self._load_path(path)
        return CovidActNowStateSummary(**data)

    def county_summary(self, fips, intervention) -> CovidActNowCountySummary:
        path = os.path.join(
            self.root, f"us/counties/{fips}.{intervention.name}.json"
        )
        data = self._load_path(path)
        return CovidActNowCountySummary(**data)

    def state_timeseries(self, state, intervention) -> CovidActNowStateTimeseries:
        path = os.path.join(
            self.root, f"us/states/{state}.{intervention.name}.timeseries.json"
        )
        data = self._load_path(path)
        return CovidActNowStateTimeseries(**data)

    def county_timeseries(self, fips, intervention) -> CovidActNowCountyTimeseries:
        path = os.path.join(
            self.root, f"us/counties/{fips}.{intervention.name}.timeseries.json"
        )
        data = self._load_path(path)
        return CovidActNowCountyTimeseries(**data)
