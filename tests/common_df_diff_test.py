import pandas as pd
from datapublic.common_fields import CommonFields
from datapublic.common_fields import FieldName

from libs import pipeline
from libs.qa import common_df_diff
from libs.qa.common_df_diff import DatasetDiff


def test_compare():
    metric_a = FieldName("metric_a")
    metric_b = FieldName("metric_b")
    columns = [CommonFields.LOCATION_ID, CommonFields.DATE, metric_a, metric_b]
    df_1 = pd.DataFrame(
        [("99", "2020-04-01", None, 1, 3), ("99", "2020-04-02", 1.1, 2.2, 3.3)],
        columns=columns + ["only_1"],
    ).set_index(common_df_diff.TIMESERIES_KEYS)

    df_2 = pd.DataFrame(
        [("99", "2020-04-01", 1, 2, 3), ("99", "2020-04-02", 1.1, None, 3.3)],
        columns=columns + ["only_2"],
    ).set_index(common_df_diff.TIMESERIES_KEYS)

    differ1 = DatasetDiff.make(df_1)
    differ2 = DatasetDiff.make(df_2)

    differ1.compare(differ2)

    assert differ1.my_ts.to_list() == [("only_1", "99")]
    assert differ2.my_ts.to_list() == [("only_2", "99")]

    assert differ1.my_ts_points.index.to_list() == [(metric_b, "99", pd.Timestamp("2020-04-02"))]
    assert differ2.my_ts_points.index.to_list() == [(metric_a, "99", pd.Timestamp("2020-04-01"))]
    assert differ1.ts_diffs.to_dict(orient="index") == {
        (metric_a, "99"): dict(diff=0, has_overlap=True, points_overlap=1),
        (metric_b, "99"): dict(diff=1 / 3, has_overlap=True, points_overlap=1),
    }


def test_drop_duplicates():
    metric_a = FieldName("metric_a")
    columns = [CommonFields.LOCATION_ID, CommonFields.DATE, metric_a]
    df_1 = pd.DataFrame(
        [("99", "2020-04-01", 1), ("99", "2020-04-01", 1.1), ("99", "2020-04-03", 3)],
        columns=columns,
    ).set_index(common_df_diff.TIMESERIES_KEYS)

    df_2 = pd.DataFrame(
        [("99", "2020-04-02", 2), ("99", "2020-04-03", 3)], columns=columns
    ).set_index(common_df_diff.TIMESERIES_KEYS)

    differ1 = DatasetDiff.make(df_1)
    differ2 = DatasetDiff.make(df_2)

    assert list(differ1.duplicates_dropped.itertuples()) == [
        (("99", "2020-04-01"), 1.0),
        (("99", "2020-04-01"), 1.1),
    ]

    differ1.compare(differ2)

    assert differ1.my_ts.to_list() == []
    assert differ2.my_ts.to_list() == []

    assert differ1.ts_diffs.to_dict(orient="index") == {
        (metric_a, "99"): dict(diff=0, has_overlap=True, points_overlap=1),
    }


def test_zero_value():
    location_id = pipeline.Region.from_state("TX").location_id
    metric_a = FieldName("metric_a")
    df_1 = pd.DataFrame(
        [(location_id, "2020-04-01", 0), (location_id, "2020-04-02", 0)],
        columns=[CommonFields.LOCATION_ID, CommonFields.DATE, metric_a],
    ).set_index(common_df_diff.TIMESERIES_KEYS)

    df_2 = df_1.copy()

    differ1 = DatasetDiff.make(df_1)
    differ2 = DatasetDiff.make(df_2)
    differ1.compare(differ2)

    assert differ1.my_ts.to_list() == []
    assert differ2.my_ts.to_list() == []

    assert differ1.ts_diffs.to_dict(orient="index") == {
        (metric_a, location_id): dict(diff=0, has_overlap=True, points_overlap=2),
    }
