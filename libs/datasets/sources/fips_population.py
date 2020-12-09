import pathlib
import pandas as pd

from covidactnow.datapublic.common_fields import CommonFields
from libs.datasets import dataset_utils
from libs.datasets import data_source
from libs.us_state_abbrev import US_STATE_ABBREV, ABBREV_US_FIPS, ABBREV_US_UNKNOWN_COUNTY_FIPS
from libs.datasets.dataset_utils import AggregationLevel

CURRENT_FOLDER = pathlib.Path(__file__).parent


class FIPSPopulation(data_source.DataSource):
    """FIPS data from US Gov census predictions + fips list.

    https://www.census.gov/data/datasets/time-series/demo/popest/2010s-counties-total.html
    https://www.census.gov/geographies/reference-files/2018/demo/popest/2018-fips.html

    Puerto Rico-specific county data:
    https://www.census.gov/data/datasets/time-series/demo/popest/2010s-total-puerto-rico-municipios.html
    """

    FILE_PATH = "data/misc/fips_population.csv"

    SOURCE_NAME = "FIPS"

    class Fields(object):
        STATE = "state"
        COUNTY = "county"
        FIPS = "fips"
        POPULATION = "population"

        # Added in standardize data.
        AGGREGATE_LEVEL = "aggregate_level"
        COUNTRY = "country"

    INDEX_FIELD_MAP = {
        CommonFields.COUNTRY: Fields.COUNTRY,
        CommonFields.STATE: Fields.STATE,
        CommonFields.FIPS: Fields.FIPS,
        CommonFields.AGGREGATE_LEVEL: Fields.AGGREGATE_LEVEL,
    }

    COMMON_FIELD_MAP = {
        CommonFields.POPULATION: Fields.POPULATION,
        CommonFields.COUNTY: CommonFields.COUNTY,  # COUNTY isn't in the LatestValueDataset.INDEX_FIELDS
    }

    def __init__(self, path):
        data = pd.read_csv(path, dtype={"fips": str})
        data["fips"] = data.fips.str.zfill(5)
        data = self.standardize_data(data)
        super().__init__(data)

    @classmethod
    def local(cls):
        data_root = dataset_utils.LOCAL_PUBLIC_DATA_PATH
        return cls(data_root / cls.FILE_PATH)

    @classmethod
    def standardize_data(cls, data: pd.DataFrame) -> pd.DataFrame:
        # Add Missing
        unknown_fips = []

        for state in data.state.unique():
            row = {
                cls.Fields.STATE: state,
                cls.Fields.FIPS: ABBREV_US_UNKNOWN_COUNTY_FIPS[state],
                cls.Fields.POPULATION: None,
                cls.Fields.COUNTY: "Unknown",
            }
            unknown_fips.append(row)

        data = data.append(unknown_fips)
        # All DH data is aggregated at the county level
        data[cls.Fields.AGGREGATE_LEVEL] = AggregationLevel.COUNTY.value
        data[cls.Fields.COUNTRY] = "USA"

        states_aggregated = dataset_utils.aggregate_and_get_nonmatching(
            data,
            [cls.Fields.COUNTRY, cls.Fields.STATE, cls.Fields.AGGREGATE_LEVEL],
            AggregationLevel.COUNTY,
            AggregationLevel.STATE,
        ).reset_index()
        states_aggregated[cls.Fields.FIPS] = states_aggregated[cls.Fields.STATE].map(ABBREV_US_FIPS)
        states_aggregated[cls.Fields.COUNTY] = None

        common_fields_data = cls._rename_to_common_fields(pd.concat([data, states_aggregated]))
        return common_fields_data


def build_fips_data_frame(census_csv, counties_csv):
    counties = pd.read_csv(counties_csv, dtype=str)
    counties.columns = [
        "summary",
        "state_fip",
        "county_fip",
        "subdivision",
        "place",
        "city",
        "name",
    ]

    county_pop = pd.read_csv(census_csv)
    county_pop.columns = ["county_state", "population"]

    # Various filters
    no_county = counties.county_fip == "000"
    has_state = counties.state_fip != "00"
    has_county = counties.county_fip != "000"
    no_subdivision = counties.subdivision == "00000"
    no_place = counties.place == "00000"
    no_city = counties.city == "00000"

    # Create state level fips
    states = counties[has_state & no_county & no_subdivision & no_city & no_place].reset_index()
    states = states.rename({"name": "state"}, axis=1)[["state_fip", "state"]]
    states.state = states.state.apply(lambda x: US_STATE_ABBREV[x])

    # Create County level
    county_only = counties[has_county & no_subdivision & no_place & no_city].reset_index()
    county_only = county_only.rename({"name": "county"}, axis=1)
    county_only["fips"] = county_only.state_fip + county_only.county_fip
    state_data = (
        county_only.set_index("state_fip")
        .join(states.set_index("state_fip"), on="state_fip")
        .reset_index()
    )

    # Sorry these lambdas are ugly
    county_pop.population = county_pop.population.apply(lambda x: int(x.replace(",", "")))
    county_pop["state"] = county_pop.county_state.apply(
        lambda x: US_STATE_ABBREV[x.split(",")[1].strip()]
    )
    county_pop["county"] = county_pop.county_state.apply(
        lambda x: x.split(",")[0].strip().lstrip(".")
    )
    county_pop = county_pop.replace("Sainte", "Ste.")
    county_pop = county_pop.replace("Saint", "St.")

    left = state_data.set_index(["state", "county"])
    right = county_pop.set_index(["state", "county"])
    results = left.join(right, on=["state", "county"]).reset_index()
    return results[["state", "county", "fips", "population"]]
