import pathlib
import numpy
import pandas as pd
from libs.datasets.population import PopulationDataset
from libs.datasets import dataset_utils
from libs.datasets import data_source
from libs.build_params import US_STATE_ABBREV
from libs import enums
CURRENT_FOLDER = pathlib.Path(__file__).parent


class PovertyPopulation(data_source.DataSource):
    """FIPS data from US Gov census predictions + fips list.
    https://www.census.gov/data-tools/demo/saipe/#/expandedTable?map_geoSelector=aa_c&s_year=2018
    """

    FILE_PATH = CURRENT_FOLDER / "poverty_census_data.csv"

    SOURCE_NAME = "POVERTY"

    class Fields(object):
        FIPS = "FIPS"
        POVERTY_PERCENT = "All Ages in Poverty Percent"

    @classmethod
    def column_mapping(cls): 
        return  {
            "County ID": cls.Fields.FIPS,
            "All Ages in Poverty Percent": cls.Fields.POVERTY_PERCENT
        }
    
    @classmethod
    def output_columns(cls): 
        return [
            cls.Fields.FIPS,
            cls.Fields.POVERTY_PERCENT
        ]

    def __init__(self, path):
        data = pd.read_csv(path, dtype={"County ID": str})
        renamed_data = data.rename(columns=PovertyPopulation.column_mapping())
        filtered_data = renamed_data.filter(PovertyPopulation.output_columns())
        filtered_data[self.Fields.FIPS] = filtered_data[self.Fields.FIPS].str.zfill(5)

        super().__init__(filtered_data)

    @classmethod
    def local(cls):
        return cls(cls.FILE_PATH)

