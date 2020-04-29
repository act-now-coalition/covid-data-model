from libs import build_processed_dataset
from libs.functions import get_can_projection


def test_get_testing_df():
    testing_df = build_processed_dataset.get_testing_timeseries_by_state('MA')
    assert testing_df is not None


def test_get_bed_for_state():
    print(get_can_projection.get_bed_data_for_state('NV'))

    assert 0
