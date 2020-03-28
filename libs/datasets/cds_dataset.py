import pandas as pd
from libs.datasets.timeseries import TimeseriesDataset
from libs.datasets import dataset_utils


def fill_missing_county_with_city(row):
    """Fills in missing county data with city if available.

    """
    if pd.isnull(row.county) and not pd.isnull(row.city):
        return row.city

    return row.county


class CDSTimeseriesData(object):
    DATA_PATH = 'data/cases-cds/timeseries.csv'

    class Fields(object):
        CITY = 'city'
        COUNTY = 'county'
        STATE = 'state'
        COUNTRY = 'country'
        POPULATION = 'population'
        LATITUDE = 'lat'
        LONGITUDE = 'long'
        URL = 'url'
        CASES = 'cases'
        DEATHS = 'deaths'
        RECOVERED = 'recovered'
        ACTIVE = 'active'
        TESTED = 'tested'
        GROWTH_FACTOR = 'growthFactor'
        DATE = 'date'

    COMMON_FIELD_MAP = {
        TimeseriesDataset.Fields.DATE: Fields.DATE,
        TimeseriesDataset.Fields.COUNTRY: Fields.COUNTRY,
        TimeseriesDataset.Fields.STATE: Fields.STATE,
        TimeseriesDataset.Fields.COUNTY: Fields.COUNTY,
        TimeseriesDataset.Fields.CASES: Fields.CASES,
        TimeseriesDataset.Fields.DEATHS: Fields.DEATHS,
        TimeseriesDataset.Fields.RECOVERED: Fields.RECOVERED,
    }

    def __init__(self, input_path):
        data = pd.read_csv(
            input_path,
            parse_dates=[self.Fields.DATE]
        )
        self.data = self.standardize_data(data)

    @classmethod
    def build_from_local_github(cls) -> 'CDSTimeseriesData':
        data_root = dataset_utils.LOCAL_PUBLIC_DATA_PATH
        return cls(data_root / cls.DATA_PATH)

    @classmethod
    def standardize_data(cls, data: pd.DataFrame) -> pd.DataFrame:
        data = dataset_utils.strip_whitespace(data)
        # TODO: Fix to use indexing better

        data[cls.Fields.COUNTY] = data.apply(fill_missing_county_with_city, axis=1)

        # Don't want to return city data because it's duplicated in county
        # City data before 3-23 was not duplicated.
        return pd.concat([
            data[data.date < '2020-03-23'],
            data[(data.date >= '2020-03-23') & data[cls.Fields.CITY].isnull()]
        ])
        return

    def to_common(self) -> TimeseriesDataset:
        to_common_fields = {
            value: key for key, value in self.COMMON_FIELD_MAP.items()
        }
        final_columns = to_common_fields.values()
        data = self.data.rename(columns=to_common_fields)[final_columns]
        data[TimeseriesDataset.Fields.SOURCE] = 'CDS'
        return TimeseriesDataset(data)

    def summarize(self, fields, country='USA'):
        pass
