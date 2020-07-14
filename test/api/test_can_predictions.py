from datetime import date
from api.can_api_definition import AggregateRegionSummaryWithTimeseries

from libs import dataset_deployer


def test_remove_root():
    value = AggregateRegionSummaryWithTimeseries(__root__=[]).dict()
    assert value == {"__root__": []}
    result = dataset_deployer.remove_root_wrapper(value)
    assert result == []
