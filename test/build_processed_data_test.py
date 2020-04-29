from libs import build_processed_dataset


def test_get_testing_df():
    testing_df = build_processed_dataset.get_testing_timeseries_by_state('MA')
    assert testing_df is not None
