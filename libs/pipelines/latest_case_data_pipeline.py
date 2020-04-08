from typing import List, Iterator, Iterable
import datetime
import json
import pathlib
import logging
from collections import namedtuple
import pydantic

from libs.datasets.timeseries import TimeseriesDataset
from libs.datasets.dataset_utils import AggregationLevel

STATE_EXPORT_FIELDS = ["state", "cases", "deaths", "source", "date"]
COUNTY_EXPORT_FIELDS = ["fips", "cases", "deaths", "source", "date"]

_logger = logging.getLogger(__name__)
LatestCasesSummary = namedtuple("LatestCasesSummary", ["state", "county"])


class StateCaseSummary(pydantic.BaseModel):
    """Case summary output in format that website expects for embeds."""

    class CountyCases(pydantic.BaseModel):
        fips: str
        cases: int
        deaths: int
        date: str

    state: str
    date: str
    cases: int
    deaths: int
    counties: List[CountyCases]


def build_summary(dataset: TimeseriesDataset) -> LatestCasesSummary:
    dataset = dataset.get_subset(None, country="USA")
    latest_state = dataset.latest_values(AggregationLevel.STATE)
    latest_county = dataset.latest_values(AggregationLevel.COUNTY)

    # I don't love this here because it's specific to the actual api output.
    latest_state["date"] = latest_state["date"].dt.strftime("%Y-%m-%d")
    latest_county["date"] = latest_county["date"].dt.strftime("%Y-%m-%d")

    return LatestCasesSummary(state=latest_state, county=latest_county)


def build_output_for_api(summary: LatestCasesSummary) -> Iterator[StateCaseSummary]:

    for state_data in summary.state.to_dict(orient="records"):
        state = state_data["state"]
        if len(state) != 2:
            _logger.info(f"Skipping state {state}")
            continue

        county_data = summary.county[summary.county.state == state]
        county_case_summaries = []
        for county in county_data.to_dict(orient="records"):
            county_summary = StateCaseSummary.CountyCases(
                fips=county[TimeseriesDataset.Fields.FIPS],
                cases=county[TimeseriesDataset.Fields.CASES],
                deaths=county[TimeseriesDataset.Fields.DEATHS],
                date=county[TimeseriesDataset.Fields.DATE],
            )
            county_case_summaries.append(county_summary)

        state_data = StateCaseSummary(
            state=state,
            date=state_data[TimeseriesDataset.Fields.DATE],
            cases=state_data[TimeseriesDataset.Fields.CASES],
            deaths=state_data[TimeseriesDataset.Fields.DEATHS],
            counties=county_case_summaries,
        )
        yield state_data


def write_output(
    output: pathlib.Path, state_summaries: Iterable[StateCaseSummary], version
):
    for state_summary in state_summaries:
        state = state_summary.state
        output_file = output / f"{state}.summary.json"
        with output_file.open("w") as f:
            _logger.info(f"Writing latest data for {state}")
            json.dump(state_summary.json(), f)

    version.write_file("case_summary", output)
