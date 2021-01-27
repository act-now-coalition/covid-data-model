from libs.datasets import data_source


class ForecastHubDataset(data_source.DataSource):
    """
    This is not yet implemented because it is not required for this dataset to be merged via the
    combined dataset pathway. Specifically, which quantiles to persist has not finalized and as such
    they are not included in CommonFields and would be filtered out regardless.
    """

    SOURCE_NAME = "ForecastHub"

    COMMON_DF_CSV_PATH = "data/forecast-hub/timeseries-common.csv"

    EXPECTED_FIELDS = [
        # Not Yet Implemented -> Currently only a move from covid-data-public to covid-data-model
    ]
