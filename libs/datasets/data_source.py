from libs.datasets.timeseries import TimeseriesDataset


class DataSource(object):

    # Subclasses must define mapping from Timeseries fields.
    # eg: {TimseriesDataset.Fields.DATE: Fields.Date}
    COMMON_FIELD_MAP = None

    # Name of dataset source.
    SOURCE_NAME = None

    def __init__(self, data):
        self.data = data

    def to_common(self, state_only=False, county_only=False):
        if state_only and county_only:
            raise ValueError("Can only specify state_only or county_only")
        data = self.data

        if state_only:
            data = data[data[self.Fields.AGGREGATE_LEVEL] == "state"]
        if county_only:
            data = data[data[self.Fields.AGGREGATE_LEVEL] == "county"]

        to_common_fields = {value: key for key, value in self.COMMON_FIELD_MAP.items()}
        final_columns = to_common_fields.values()
        data = data.rename(columns=to_common_fields)[final_columns]
        data[TimeseriesDataset.Fields.SOURCE] = self.SOURCE_NAME

        return TimeseriesDataset(data)

    @classmethod
    def build_from_local_github(cls) -> "cls":
        """Builds data from local covid-public-data github repo.

        Returns: Instantiated class with data loaded.
        """
        raise NotImplementedError("Subclass must implement")
