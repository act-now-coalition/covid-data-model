import logging
import pytest
import pandas as pd
from libs.datasets import JHUDataset
from libs.datasets import CDSDataset
from libs.datasets import dataset_utils
from libs.datasets import custom_aggregations
from libs.datasets.dataset_utils import AggregationLevel
from libs import CovidDatasets

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


def default_timeseries_row(**updates):
    data = {
        'fips': '22083',
        'aggregate_level': 'county',
        'date': '2020-03-26 00:00:00',
        'country': 'USA',
        'state': 'LA',
        'cases': 10.0,
        'deaths': 1.0,
        'recovered': 0.0,
        'source': 'JHU',
        'generated': False,
        'county': 'Richland Parish'
    }
    data.update(updates)
    return data


@pytest.mark.parametrize("are_boroughs_zero", [True, False])
def test_nyc_aggregation(are_boroughs_zero):

    nyc_county_fips = custom_aggregations.NEW_YORK_COUNTY_FIPS
    nyc_borough_fips = custom_aggregations.NYC_BOROUGH_FIPS[0]

    nyc_cases = 10
    borough_cases = 0 if are_boroughs_zero else 10
    rows = [
        default_timeseries_row(fips=nyc_county_fips, cases=nyc_cases),
        default_timeseries_row(
            fips=nyc_borough_fips, cases=borough_cases, deaths=borough_cases, recovered=borough_cases
        ),
        default_timeseries_row()
    ]

    df = pd.DataFrame(rows)

    # Todo: figure out a better way to define these groups.
    group = [
        'date', 'source', 'country', 'aggregate_level', 'state', 'generated'
    ]
    result = custom_aggregations.update_with_combined_new_york_counties(
        df, group, are_boroughs_zero=are_boroughs_zero
    )
    results = result.sort_values('fips').to_dict(orient='records')

    assert len(results) == 2
    nyc_result = results[1]

    if are_boroughs_zero:
        assert nyc_result['cases'] == nyc_cases
    else:
        assert nyc_result['cases'] == nyc_cases + borough_cases
