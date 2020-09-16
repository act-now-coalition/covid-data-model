"""
Code that is used to help move information around in the pipeline, starting with `Region` which
represents a geographical area (state, county, metro area, etc).
"""

# Many other modules import this module. Importing pyseir or dataset code here is likely to create
# in import cycle.
import warnings
from dataclasses import dataclass
from typing import Optional

import us
from typing_extensions import final


class BadFipsWarning(UserWarning):
    pass


def fips_to_location_id(fips: str) -> str:
    """Converts a FIPS code to a locationID"""
    state_obj = us.states.lookup(fips[0:2], field="fips")
    if state_obj:
        if len(fips) == 2:
            return f"iso1:us#iso2:us-{state_obj.abbr.lower()}"
        elif len(fips) == 5:
            return f"iso1:us#iso2:us-{state_obj.abbr.lower()}#fips:{fips}"

    warnings.warn(BadFipsWarning(f"Fallback locationID for fips {fips}"), stacklevel=2)
    return f"iso1:us#fips:{fips}"


def cbsa_to_location_id(cbsa_code: str) -> str:
    return f"iso1:us#cbsa:{cbsa_code}"


@final
@dataclass(frozen=True)
class Region:
    """Identifies a geographical area."""

    # locationID in the style of CovidAtlas/Project Li. See
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
        return Region(location_id=cbsa_to_location_id(cbsa_code), fips=None)

    def is_county(self):
        return len(self.fips) == 5

    def is_state(self):
        return len(self.fips) == 2

    def state_obj(self):
        if self.is_state():
            return us.states.lookup(self.fips)
        elif self.is_county():
            return us.states.lookup(self.fips[:2])
        else:
            raise ValueError(f"No state_obj for {self}")

    def get_state_region(self) -> "Region":
        """Returns a Region object for the state of a county, otherwise raises a ValueError."""
        if len(self.fips) != 5:
            raise ValueError(f"No state for {self}")
        return Region.from_fips(self.fips[:2])
