import pathlib

import moto
import boto3
import pytest

from libs.datasets import combined_dataset_utils
from libs.datasets.combined_dataset_utils import DatasetType
from libs.datasets import combined_datasets
from libs.qa.common_df_diff import DatasetDiff


def test_persist_and_load_dataset(tmp_path, nyc_fips):
    dataset = combined_datasets.load_us_timeseries_dataset()
    timeseries_nyc = dataset.get_subset(None, fips=nyc_fips)

    pointer = combined_dataset_utils.persist_dataset(timeseries_nyc, tmp_path)

    downloaded_dataset = pointer.load_dataset()
    differ_l = DatasetDiff.make(downloaded_dataset.data)
    differ_r = DatasetDiff.make(timeseries_nyc.data)
    differ_l.compare(differ_r)

    assert not len(differ_l.my_ts)


def test_update_and_load(tmp_path: pathlib.Path, nyc_fips):
    latest = combined_datasets.load_us_latest_dataset()
    timeseries_dataset = combined_datasets.load_us_timeseries_dataset()

    # restricting the datasets being persisted to one county to speed up tests a bit.
    latest_nyc = latest.get_subset(None, fips=nyc_fips)
    timeseries_nyc = timeseries_dataset.get_subset(None, fips=nyc_fips)

    combined_dataset_utils.update_data_public_head(
        tmp_path, latest_dataset=latest_nyc, timeseries_dataset=timeseries_nyc,
    )

    timeseries = combined_datasets.load_us_timeseries_dataset(pointer_directory=tmp_path)

    latest = combined_datasets.load_us_latest_dataset(pointer_directory=tmp_path)

    assert latest.get_record_for_fips(nyc_fips) == latest_nyc.get_record_for_fips(nyc_fips)
