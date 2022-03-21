"""
Code that is used to help move information around in the pipeline, starting with `Region` which
represents a geographical area (state, county, metro area, etc).
"""

# Many other modules import this module. Importing pyseir or dataset code here is likely to create
# in import cycle.
import re
import warnings
from dataclasses import dataclass
from typing import List
from typing import Mapping
from typing import Optional
from typing import Union

import us
import pandas as pd
from datapublic.common_fields import CommonFields
from typing_extensions import NewType
from typing_extensions import final

from libs import us_state_abbrev
from libs.datasets.dataset_utils import AggregationLevel
from libs.datasets import dataset_utils


class BadFipsWarning(UserWarning):
    pass


def fips_to_location_id(fips: str) -> str:
    """Converts a FIPS code to a location_id"""
    try:
        return dataset_utils.get_fips_to_location().at[fips]
    except KeyError:
        # This happens mostly (entirely?) in unittest data where the first two digits
        # are not a valid state FIPS. See
        # https://trello.com/c/QEbSwjSQ/631-remove-support-for-county-locationid-without-state
        warnings.warn(BadFipsWarning(f"Fallback location_id for fips {fips}"), stacklevel=2)
        return f"iso1:us#fips:{fips}"


def location_id_to_fips(location_id: str) -> Optional[str]:
    """Converts a location_id to a FIPS code"""
    return dataset_utils.get_geo_data().at[location_id, CommonFields.FIPS]


def location_id_to_level(location_id: str) -> Optional[AggregationLevel]:
    """Converts a location_id to a FIPS code"""
    match = re.fullmatch(r"iso1:us#.*fips:(\d+)", location_id)
    if match:
        fips = match.group(1)
        if len(fips) == 2:
            return AggregationLevel.STATE
        elif len(fips) == 5:
            return AggregationLevel.COUNTY
        elif len(fips) == 7:
            return AggregationLevel.PLACE

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
    https://github.com/covid-projections/covid-data-public/tree/main/data/census-msa
    """
    return f"iso1:us#cbsa:{cbsa_code}"


def hsa_to_location_id(hsa_code: str) -> str:
    """Turns a HSA code into a location_id.

    For information about how these identifiers are brought into the CAN code see
    https://github.com/covid-projections/covid-data-public/tree/main/data/census-msa
    """
    return f"iso1:us#hsa:{hsa_code}"


@final
@dataclass(frozen=True)
class Region:
    """Identifies a geographical area."""

    # In the style of CovidAtlas/Project Li `locationID`. See
    # https://github.com/covidatlas/li/blob/master/docs/reports-v1.md#general-notes
    location_id: str

    # The FIPS identifier for the region, either 2 digits for a state, 5 digits for a county or 7
    # digits for a place.
    fips: Optional[str]

    @staticmethod
    def from_fips(fips: str) -> "Region":
        """Creates a Region object from a state, county or place FIPS code.

        Use from_cbsa_code for CBSAs; this function assumes fips[0:2] is the correct state code."""
        return Region(location_id=fips_to_location_id(fips), fips=fips)

    @staticmethod
    def from_state(state: str) -> "Region":
        """Creates a Region object from a state abbreviation, name or 2 digit FIPS code."""
        state_obj = us.states.lookup(state)
        fips = state_obj.fips
        return Region.from_fips(fips)

    @staticmethod
    def from_cbsa_code(cbsa_code: str) -> "Region":
        """Creates a Region object from a CBSA FIPS code.

        Use from_fips for state, county or place FIPS."""
        fips = cbsa_code
        return Region(location_id=cbsa_to_location_id(cbsa_code), fips=fips)

    @staticmethod
    def from_hsa_code(hsa_code: str) -> "Region":
        """Creates a Region object from a HSA code.

        Use from_fips for state, county or place FIPS."""
        fips = hsa_code
        return Region(location_id=hsa_to_location_id(hsa_code), fips=fips)

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
    def country(self) -> str:
        """2-letter ISO-3166 Country code."""
        # TODO(chris): Make more generic if we want to support other countries.
        return "US"

    @property
    def level(self) -> AggregationLevel:
        level = location_id_to_level(self.location_id)

        if level:
            return level

        raise NotImplementedError("Unknown Aggregation Level")

    @property
    def fips_for_api(self) -> str:
        """The same as `fips`, except '0' for the USA as a hack to help the frontend."""
        if self.level is AggregationLevel.COUNTRY:
            assert self.location_id == "iso1:us"
            return "0"
        else:
            return self.fips

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
        """Returns a Region object for the state of a county or place, otherwise raises a
        ValueError."""
        if len(self.fips) != 5 and len(self.fips) != 7:
            raise ValueError(f"No state for {self}")
        return Region.from_fips(self.fips[:2])


@final
@dataclass(frozen=True)
class RegionMask:
    """Represents attributes which may be used to select a subset of regions."""

    # A level (county, state, ...) OR None to select regions ignoring their level.
    level: Optional[AggregationLevel] = None
    # A list of states, each a two letter string OR None to select regions ignoring their state.
    states: Optional[List[str]] = None


RegionMaskOrRegion = NewType("RegionMaskOrRegion", Union[RegionMask, Region])


def us_states_and_territories_to_country_map() -> Mapping[Region, Region]:
    us_country_region = Region.from_location_id("iso1:us")
    return {
        Region.from_state(state): us_country_region
        for state in us_state_abbrev.US_STATE_ABBREV.values()
    }
