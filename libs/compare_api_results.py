from typing import Optional
import os
import json
import requests
import pandas as pd
import logging
import itertools
from libs.enums import Intervention
from libs import us_state_abbrev
from api.can_api_definition import CovidActNowStateSummary
from api.can_api_definition import CovidActNowCountySummary
from api.can_api_definition import CovidActNowStateTimeseries
from api.can_api_definition import CovidActNowCountyTimeseries


_logger = logging.getLogger(__name__)

CAN_API_BASE_URL = "https://data.covidactnow.org/"


class APISnapshot(object):
    def __init__(self, root: str):
        self.root = root
        self._is_url = self.root.startswith("https://")

    def _load_path(self, path):
        try:
            if self._is_url:
                return requests.get(path).json()

            with open(path) as f:
                return json.load(f)
        except Exception:
            _logger.warning(f"Failed to load {path}")
            return None

    @classmethod
    def from_snapshot(cls, snapshot_id: int):
        path = os.path.join(CAN_API_BASE_URL, f"snapshot/{snapshot_id}")
        return cls(path)

    @classmethod
    def from_latest(cls):
        path = os.path.join(CAN_API_BASE_URL, f"latest")
        return cls(path)

    def state_summary(self, state, intervention) -> Optional[CovidActNowStateSummary]:
        path = os.path.join(self.root, f"us/states/{state}.{intervention.name}.json")
        data = self._load_path(path)
        if not data:
            return None
        return CovidActNowStateSummary(**data)

    def county_summary(self, fips, intervention) -> Optional[CovidActNowCountySummary]:
        path = os.path.join(self.root, f"us/counties/{fips}.{intervention.name}.json")
        data = self._load_path(path)
        if not data:
            return None
        return CovidActNowCountySummary(**data)

    def state_timeseries(
        self, state, intervention
    ) -> Optional[CovidActNowStateTimeseries]:
        path = os.path.join(
            self.root, f"us/states/{state}.{intervention.name}.timeseries.json"
        )
        data = self._load_path(path)
        if not data:
            return None
        return CovidActNowStateTimeseries(**data)

    def county_timeseries(
        self, fips, intervention
    ) -> Optional[CovidActNowCountyTimeseries]:
        path = os.path.join(
            self.root, f"us/counties/{fips}.{intervention.name}.timeseries.json"
        )
        data = self._load_path(path)
        if not data:
            return None
        return CovidActNowCountyTimeseries(**data)


def get_actual_projection_deltas(summary_with_timeseries, intervention):
    date = summary_with_timeseries.lastUpdatedDate

    actuals = summary_with_timeseries.actuals
    matching_projection_rows = [
        row for row in summary_with_timeseries.timeseries if row.date == date
    ]
    if len(matching_projection_rows) > 1 or not matching_projection_rows:
        raise ValueError(f"Single row not found {date}")

    projection_row = matching_projection_rows[0]
    common = {
        "date": date,
        "intervention": intervention.name,
        "state": summary_with_timeseries.stateName,
        "fips": summary_with_timeseries.fips,
    }
    beds_in_use = {
        **common,
        "field": "hospital_beds",
        "actual": actuals.hospitalBeds.currentUsage,
        "projection": projection_row.hospitalBedsRequired,
    }
    icu_in_use = {
        **common,
        "field": "icu_in_use",
        "projection": projection_row.ICUBedsInUse,
    }
    if actuals.ICUBeds:
        icu_in_use["actual"] = actuals.ICUBeds.currentUsage

    return [beds_in_use, icu_in_use]


def build_comparison_for_states(api_snapshot: APISnapshot) -> pd.DataFrame:
    """Loads all state timeseries summaries and builds comparison of actuals to predictions.

    Args:
        api_snapshot: API Snapshot object

    Returns: Pandas comparison data frame.
    """
    all_comparisons = []

    states = us_state_abbrev.abbrev_us_state.keys()
    interventions = list(Intervention)
    states_and_interventions = list(itertools.product(states, interventions))

    def get_results(state, intervention):
        # intervention is not included in the summary result, so we need to
        # pass back which intervention this is for.
        result = api_snapshot.state_timeseries(state, intervention)
        return state, intervention, result

    summaries = [
        get_results(state, intervention)
        for state, intervention in states_and_interventions
    ]

    for state, intervention, summary in summaries:
        if not summary:
            continue

        values = get_actual_projection_deltas(summary, intervention)
        all_comparisons.extend(values)

    return pd.DataFrame(all_comparisons)
