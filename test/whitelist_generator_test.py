import pytest

from pyseir.inference.whitelist_generator import WhitelistGenerator


# turns all warnings into errors for this module
pytestmark = pytest.mark.filterwarnings("error")


def test_all_data_smoke_test():
    df = WhitelistGenerator().generate_whitelist()
    assert df
