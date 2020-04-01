from libs.datasets import JHUDataset
from libs.datasets import CDSDataset
from libs.datasets import dataset_utils
from libs.datasets.dataset_utils import AggregationLevel
from libs import CovidDatasets
import pytest
import logging

_logger = logging.getLogger(__name__)

@pytest.mark.skip
@pytest.mark.parametrize(
    "test_missing,test_matching",
    [(True, True)]
)
@pytest.mark.parametrize(
    "legacy_cls,new_cls",
    [(CovidDatasets.JHUDataset, JHUDataset), (CovidDatasets.CDSDataset, CDSDataset),],
)
def test_missing_state_in_generic_dataset(legacy_cls, new_cls, test_matching, test_missing):
    if test_matching:
        test_type = "values match"
    elif test_missing:
        test_type = "values are both included"

    print(f"Running on {legacy_cls} checking that {test_type}")
    legacy_jhu = legacy_cls()
    jhu = new_cls.local().timeseries().get_subset(None, after="2020-03-02")
    new = jhu.get_subset(AggregationLevel.STATE, country="USA")

    new.latest_values(AggregationLevel.STATE)
    state_groupby = ["country", "date", "state"]
    not_matching_states = []
    missing_states = []
    for state in new.states:
        if len(state) > 2:
            # Some of the states have weird data (i.e. cruise ship), skipping
            continue
        try:
            old_timeseries = legacy_jhu.get_timeseries_by_country_state(
                "USA", state, 4
            )
        except Exception:
            print(f"missing data for old timeseries: {state}")
        new_timeseries = new.get_data(state=state)

        # New data does not contain synthetics.
        non_synthetic = old_timeseries[old_timeseries.synthetic.isnull()]
        comparison_result = dataset_utils.compare_datasets(
            non_synthetic,
            new_timeseries,
            state_groupby,
            first_name="old",
            other_name="new",
        )
        all_combined, matching, not_matching, missing = comparison_result

        if test_matching and len(not_matching):
            not_matching_states.append((state, not_matching))

        if test_missing and len(missing):
            missing_states.append((state, missing))

    if not_matching_states:
        for state, data in not_matching_states:
            print(state)
            print(data)

    if missing_states:
        for state, data in missing_states:
            print(state)
            print(data)

    if not_matching_states or missing_states:
        assert False


def test_aggregation():
    pass
