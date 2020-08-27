import pytest

from libs.datasets import combined_datasets
from pyseir.inference.whitelist_generator import WhitelistGenerator


# turns all warnings into errors for this module
pytestmark = pytest.mark.filterwarnings("error")


def test_all_data_smoke_test():
    input_timeseries = combined_datasets.load_us_timeseries_dataset().get_subset(state="AL")
    df = WhitelistGenerator().generate_whitelist(input_timeseries)
    assert not df.empty
