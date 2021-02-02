import more_itertools
import structlog
from covidactnow.datapublic.common_fields import CommonFields

from libs.datasets.sources import zeros_filter
from libs.pipeline import Region
from tests import test_helpers
from tests.test_helpers import TimeseriesLiteral
import pandas as pd


def test_basic():
    region_tx = Region.from_state("TX")
    region_sf = Region.from_fips("06075")
    region_hi = Region.from_state("HI")
    # Add a timeseries with a tag to make sure they are preserved.
    ts_with_tag = TimeseriesLiteral(
        [0, 0, 0], annotation=[test_helpers.make_tag(date="2020-04-01")]
    )
    ds_in = test_helpers.build_dataset(
        {
            region_tx: {CommonFields.VACCINES_DISTRIBUTED: [0, 0, 0]},
            region_sf: {CommonFields.VACCINES_DISTRIBUTED: [0, 0, 1]},
            region_hi: {
                CommonFields.VACCINES_DISTRIBUTED: [0, 0, None],
                CommonFields.CASES: ts_with_tag,
            },
        }
    )

    with structlog.testing.capture_logs() as logs:
        ds_out = zeros_filter.drop_all_zero_timeseries(ds_in, [CommonFields.VACCINES_DISTRIBUTED])
    ds_expected = test_helpers.build_dataset(
        {
            region_sf: {CommonFields.VACCINES_DISTRIBUTED: [0, 0, 1]},
            region_hi: {CommonFields.CASES: ts_with_tag},
        }
    )
    log = more_itertools.one(logs)
    assert log["event"] == zeros_filter.DROPPING_TIMESERIES_WITH_ONLY_ZEROS
    assert pd.MultiIndex.from_tuples(
        [
            (region_hi.location_id, CommonFields.VACCINES_DISTRIBUTED),
            (region_tx.location_id, CommonFields.VACCINES_DISTRIBUTED),
        ]
    ).equals(log["dropped"])
    test_helpers.assert_dataset_like(ds_expected, ds_out)
