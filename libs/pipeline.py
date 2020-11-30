"""
Code that is used to help move information around in the pipeline, starting with `Region` which
represents a geographical area (state, county, metro area, etc).
"""

# Many other modules import this module. Importing pyseir or dataset code here is likely to create
# in import cycle.
import re
import warnings
from dataclasses import dataclass
from typing import Mapping
from typing import Optional

import us
from typing_extensions import final

from libs import us_state_abbrev
from libs.datasets.dataset_utils import AggregationLevel


class BadFipsWarning(UserWarning):
    pass


def fips_to_location_id(fips: str) -> str:
    """Converts a FIPS code to a location_id"""
    state_obj = us.states.lookup(fips[0:2], field="fips")
    if state_obj:
        if len(fips) == 2:
            return f"iso1:us#iso2:us-{state_obj.abbr.lower()}"
        elif len(fips) == 5:
            return f"iso1:us#iso2:us-{state_obj.abbr.lower()}#fips:{fips}"

    # This happens mostly (entirely?) in unittest data where the first two digits
    # are not a valid state FIPS. See
    # https://trello.com/c/QEbSwjSQ/631-remove-support-for-county-locationid-without-state
    warnings.warn(BadFipsWarning(f"Fallback location_id for fips {fips}"), stacklevel=2)
    return f"iso1:us#fips:{fips}"


def location_id_to_fips(location_id: str) -> Optional[str]:
    """Converts a location_id to a FIPS code"""
    match = re.fullmatch(r"iso1:us#.*fips:(\d+)", location_id)
    if match:
        return match.group(1)

    match = re.fullmatch(r"iso1:us#iso2:us-(..)", location_id)
    if match:
        return us.states.lookup(match.group(1).upper(), field="abbr").fips

    match = re.fullmatch(r"iso1:us#cbsa:(\d+)", location_id)
    if match:
        return match.group(1)

    return None


def location_id_to_level(location_id: str) -> Optional[AggregationLevel]:
    """Converts a location_id to a FIPS code"""
    match = re.fullmatch(r"iso1:us#.*fips:(\d+)", location_id)
    if match:
        fips = match.group(1)
        if len(fips) == 2:
            return AggregationLevel.STATE
        if len(fips) == 5:
            return AggregationLevel.COUNTY

    match = re.fullmatch(r"iso1:us#iso2:us-(..)", location_id)
    if match:
        return AggregationLevel.STATE

    match = re.fullmatch(r"iso1:us#cbsa:(\d+)", location_id)
    if match:
        return AggregationLevel.CBSA

    match = re.fullmatch(r"iso1:\w\w", location_id)
    if match:
        return AggregationLevel.COUNTRY

    return None


def cbsa_to_location_id(cbsa_code: str) -> str:
    """Turns a CBSA code into a location_id.

    For information about how these identifiers are brought into the CAN code see
    https://github.com/covid-projections/covid-data-public/tree/master/data/census-msa
    """
    return f"iso1:us#cbsa:{cbsa_code}"


@final
@dataclass(frozen=True)
class Region:
    """Identifies a geographical area."""

    # In the style of CovidAtlas/Project Li `locationID`. See
    # https://github.com/covidatlas/li/blob/master/docs/reports-v1.md#general-notes
    location_id: str

    # The FIPS identifier for the region, either 2 digits for a state or 5 digits for a county.
    fips: Optional[str]

    @staticmethod
    def from_fips(fips: str) -> "Region":
        return Region(location_id=fips_to_location_id(fips), fips=fips)

    @staticmethod
    def from_state(state: str) -> "Region":
        """Creates a Region object from a state abbreviation, name or 2 digit FIPS code."""
        state_obj = us.states.lookup(state)
        fips = state_obj.fips
        return Region.from_fips(fips)

    @staticmethod
    def from_cbsa_code(cbsa_code: str) -> "Region":
        # cbsa_code is a valid fips code, setting it to fips code
        fips = cbsa_code
        return Region(location_id=cbsa_to_location_id(cbsa_code), fips=fips)

    @staticmethod
    def from_location_id(location_id: str) -> "Region":
        fips = location_id_to_fips(location_id)
        return Region(location_id=location_id, fips=fips)

    @staticmethod
    def from_iso1(iso1: str) -> "Region":
        assert len(iso1) == 2
        assert iso1 == "us"  # Remove when we start supporting other countries :-)
        return Region(location_id=f"iso1:{iso1}", fips=None)

    @property
    def state(self) -> Optional[str]:
        state_obj = self.state_obj()
        if state_obj:
            return state_obj.abbr

        return None

    @property
    def country(self):
        return "USA"

    @property
    def level(self) -> AggregationLevel:
        level = location_id_to_level(self.location_id)

        if level:
            return level

        raise NotImplementedError("Unknown Aggregation Level")

    def is_county(self):
        return self.level is AggregationLevel.COUNTY

    def is_state(self):
        return self.level is AggregationLevel.STATE

    def state_obj(self):
        if self.is_state():
            return us.states.lookup(self.fips)
        elif self.is_county():
            return us.states.lookup(self.fips[:2])

        return None

    def get_state_region(self) -> "Region":
        """Returns a Region object for the state of a county, otherwise raises a ValueError."""
        if len(self.fips) != 5:
            raise ValueError(f"No state for {self}")
        return Region.from_fips(self.fips[:2])


def us_states_to_country_map() -> Mapping[Region, Region]:
    us_country_region = Region.from_location_id("iso1:us")
    # Sorry US Territories, only including 50 states and DC for now.
    return {
        Region.from_state(state): us_country_region for state in us_state_abbrev.STATES_50.values()
    }
