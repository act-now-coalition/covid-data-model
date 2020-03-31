class DataSource(object):

    # Subclasses must define mapping from Timeseries fields.
    # eg: {TimseriesDataset.Fields.DATE: Fields.Date}
    TIMESERIES_FIELD_MAP = None

    BEDS_FIELD_MAP = None

    # Name of dataset source.
    SOURCE_NAME = None

    def __init__(self, data):
        self.data = data

    @classmethod
    def build_from_local_github(cls) -> "cls":
        """Builds data from local covid-public-data github repo.

        Returns: Instantiated class with data loaded.
        """
        raise NotImplementedError("Subclass must implement")

    def to_generic_beds(self) -> "BedsDataset":
        """Builds generic beds dataset"""
        from libs.datasets.beds import BedsDataset

        return BedsDataset.from_source(self)

    def to_generic_population(self) -> "PopulationDataset":
        """Builds generic beds dataset"""
        from libs.datasets.population import PopulationDataset

        return PopulationDataset.from_source(self)

    def to_generic_timeseries(self) -> "TimeseriesDataset":
        """Builds generic beds dataset"""
        from libs.datasets.timeseries import TimeseriesDataset

        return TimeseriesDataset.from_source(self)
