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
from api.can_api_definition import CovidActNowStatesTimeseries
from api.can_api_definition import CovidActNowCountiesTimeseries
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

    def all_states_timeseries(self, intervention):
        path = os.path.join(
            self.root, f"us/states.{intervention.name}.timeseries.json"
        )
        data = self._load_path(path)
        if not data:
            return None
        return CovidActNowStatesTimeseries(__root__=data)

    def all_counties_timeseries(self, intervention):
        path = os.path.join(
            self.root, f"us/counties.{intervention.name}.timeseries.json"
        )
        data = self._load_path(path)
        if not data:
            return None
        return CovidActNowCountiesTimeseries(__root__=data)

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
        "actual": actuals.hospitalBeds.currentUsageCovid,
        "projection": projection_row.hospitalBedsRequired,
    }
    icu_in_use = {
        **common,
        "field": "icu_in_use",
        "projection": projection_row.ICUBedsInUse,
    }
    if actuals.ICUBeds:
        icu_in_use["actual"] = actuals.ICUBeds.currentUsageCovid

    return [beds_in_use, icu_in_use]


def add_comparisons(comparisons):
    # Hacky lets set this
    comparisons['actual'] = comparisons['actual'].astype(float)
    comparisons['projection'] = comparisons['projection'].astype(float)
    comparisons['delta'] = comparisons['projection'] - comparisons['actual']
    comparisons['delta_ratio'] = comparisons['delta'] / comparisons['actual']
    return comparisons


def build_comparison_for_states(api_snapshot: APISnapshot, deduplicate=True) -> pd.DataFrame:
    """Loads all state timeseries summaries and builds comparison of actuals to predictions.

    Args:
        api_snapshot: API Snapshot object
        deduplicate: If true deduplicates data

    Returns: Pandas comparison data frame.
    """
    all_comparisons = []

    summaries = [
        (api_snapshot.all_states_timeseries(intervention), intervention)
        for intervention in Intervention
    ]

    for summary, intervention in summaries:
        for state_summary in summary.__root__:
            values = get_actual_projection_deltas(state_summary, intervention)
            all_comparisons.extend(values)

    data = pd.DataFrame(all_comparisons)
    if deduplicate:
        data = data.set_index(['date', 'state', 'field'])
        data = data[~data.index.duplicated(keep='last')].drop(columns=['intervention'], axis=1)

    data = add_comparisons(data)
    return data



def build_comparison_for_states(api_snapshot: APISnapshot, deduplicate=True) -> pd.DataFrame:
    """Loads all state timeseries summaries and builds comparison of actuals to predictions.

    Args:
        api_snapshot: API Snapshot object
        deduplicate: If true deduplicates data

    Returns: Pandas comparison data frame.
    """
    all_comparisons = []

    summaries = [
        (api_snapshot.all_states_timeseries(intervention), intervention)
        for intervention in Intervention
    ]

    for summary, intervention in summaries:
        for state_summary in summary.__root__:
            values = get_actual_projection_deltas(state_summary, intervention)
            all_comparisons.extend(values)

    data = pd.DataFrame(all_comparisons)
    if deduplicate:
        data = data.set_index(['date', 'state', 'field'])
        data = data[~data.index.duplicated(keep='last')].drop(columns=['intervention'], axis=1)

    data = add_comparisons(data)
    return data
