from pyseir.inference.whitelist_generator import WhitelistGenerator


def test_all_data_smoke_test():
    df = WhitelistGenerator().generate_whitelist()
    assert df
