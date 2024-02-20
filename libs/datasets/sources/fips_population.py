import pathlib
from functools import lru_cache

import pandas as pd

from datapublic.common_fields import CommonFields
from libs.datasets import dataset_utils
from libs.datasets import data_source
from libs.datasets import timeseries
from libs.us_state_abbrev import ABBREV_US_FIPS, ABBREV_US_UNKNOWN_COUNTY_FIPS
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

    SOURCE_TYPE = "FIPS"

    class Fields(object):
        STATE = "state"
        COUNTY = "county"
        FIPS = "fips"
        POPULATION = "population"

        # Added in standardize data.
        AGGREGATE_LEVEL = "aggregate_level"
        COUNTRY = "country"

    EXPECTED_FIELDS = [CommonFields.POPULATION, CommonFields.COUNTY]

    @classmethod
    @lru_cache(None)
    def make_dataset(cls) -> timeseries.MultiRegionDataset:
        data_root = dataset_utils.REPO_ROOT
        data = pd.read_csv(data_root / cls.FILE_PATH, dtype={"fips": str})
        data["fips"] = data.fips.str.zfill(5)
        data = cls.standardize_data(data)
        return timeseries.MultiRegionDataset.new_without_timeseries().add_fips_static_df(data)

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

        data = pd.concat([data, pd.DataFrame.from_records(unknown_fips)])
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

        us_row = {
            CommonFields.FIPS: "0",
            CommonFields.COUNTRY: "USA",
            CommonFields.POPULATION: states_aggregated[CommonFields.POPULATION].sum(),
        }
        country_aggregated = pd.DataFrame.from_records([us_row])

        common_fields_data = pd.concat([data, states_aggregated, country_aggregated])
        return common_fields_data
