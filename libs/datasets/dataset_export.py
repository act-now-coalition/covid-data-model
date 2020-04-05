from typing import Iterator
import logging

from libs.datasets.timeseries import TimeseriesDataset
from libs.datasets.dataset_utils import AggregationLevel


_logger = logging.getLogger(__name__)

STATE_EXPORT_FIELDS = ["state", "cases", "deaths", "source", "date"]
COUNTY_EXPORT_FIELDS = ["fips", "cases", "deaths", "source", "date"]


def latest_case_summaries_by_state(dataset: TimeseriesDataset) -> Iterator[dict]:
    """Builds summary of latest case data by state and county.

    Data is generated for the embeds which expects a list of records in this format:
    {
        "state": <state>,
        "date": "YYYY-MM-DD",
        "cases": <cases>,
        "deaths": <deaths>,
        "counties": [
            {"fips": <fips code>, "cases": <cases>, "deaths": <deaths", "date": <date>}
        ]
    }

    Args:
        data: Timeseries object.

    Returns: List of data.
    """

    dataset = dataset.get_subset(None, country="USA")
    latest_state = dataset.latest_values(AggregationLevel.STATE)
    latest_county = dataset.latest_values(AggregationLevel.COUNTY)

    latest_state["date"] = latest_state["date"].dt.strftime("%Y-%m-%d")
    latest_county["date"] = latest_county["date"].dt.strftime("%Y-%m-%d")

    states = latest_state[STATE_EXPORT_FIELDS].to_dict(orient="records")

    for state_data in states:
        state = state_data["state"]
        if len(state) != 2:
            _logger.info(f"Skipping state {state}")
            continue

        county_data = latest_county[latest_county.state == state]
        counties = county_data[COUNTY_EXPORT_FIELDS].to_dict(orient="records")

        state_data.update({"counties": counties})
        yield state, state_data
