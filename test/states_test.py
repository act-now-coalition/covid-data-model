from libs.datasets.sources.states import StatesData


def test_loads_with_default_parameters():
    data = StatesData.local()
    assert not data.data.empty
