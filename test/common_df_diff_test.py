import pandas as pd

from libs.qa.common_df_diff import DatasetDiff
from covidactnow.datapublic.common_fields import COMMON_FIELDS_TIMESERIES_KEYS


def test_compare():
    df_1 = pd.DataFrame(
        [("99", "2020-04-01", None, 1, 3), ("99", "2020-04-02", 1.1, 2.2, 3.3)],
        columns="fips date metric_a metric_b only_1".split(),
    ).set_index(COMMON_FIELDS_TIMESERIES_KEYS)

    df_2 = pd.DataFrame(
        [("99", "2020-04-01", 1, 2, 3), ("99", "2020-04-02", 1.1, None, 3.3)],
        columns="fips date metric_a metric_b only_2".split(),
    ).set_index(COMMON_FIELDS_TIMESERIES_KEYS)

    differ1 = DatasetDiff.make(df_1)
    differ2 = DatasetDiff.make(df_2)

    differ1.compare(differ2)

    assert differ1.my_ts.to_list() == [("only_1", "99")]
    assert differ2.my_ts.to_list() == [("only_2", "99")]

    assert differ1.my_ts_points.index.to_list() == [("metric_b", "99", "2020-04-02")]
    assert differ2.my_ts_points.index.to_list() == [("metric_a", "99", "2020-04-01")]
    assert differ1.ts_diffs.to_dict(orient="index") == {
        ("metric_a", "99"): dict(diff=0, has_overlap=True, points_overlap=1),
        ("metric_b", "99"): dict(diff=1 / 1.5, has_overlap=True, points_overlap=1),
    }
