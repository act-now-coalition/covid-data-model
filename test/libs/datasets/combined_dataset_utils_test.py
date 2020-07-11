import pathlib

import moto
import boto3
import pytest

from libs.datasets import combined_dataset_utils
from libs.datasets.combined_dataset_utils import DatasetType
from libs.datasets.combined_dataset_utils import DatasetPromotion
from libs.datasets import combined_datasets
from libs.qa.common_df_diff import DatasetDiff


@pytest.fixture
def mock_s3_bucket():
    with moto.mock_s3():
        s3 = boto3.resource("s3", region_name="us-east-1")
        bucket = "test-bucket"
        s3.create_bucket(Bucket=bucket)
        yield bucket


def test_persist_and_load_dataset(mock_s3_bucket: str, tmp_path, nyc_fips):
    s3_prefix = f"s3://{mock_s3_bucket}"
    dataset = combined_datasets.build_us_timeseries_with_all_fields()
    timeseries_nyc = dataset.get_subset(None, fips=nyc_fips)

    pointer = combined_dataset_utils.persist_dataset(timeseries_nyc, s3_prefix)

    downloaded_dataset = pointer.load_dataset(download_directory=tmp_path)
    differ_l = DatasetDiff.make(downloaded_dataset.data)
    differ_r = DatasetDiff.make(timeseries_nyc.data)
    differ_l.compare(differ_r)

    assert not len(differ_l.my_ts)


def test_update_and_load(mock_s3_bucket: str, tmp_path: pathlib.Path, nyc_fips):
    latest = combined_datasets.build_us_latest_with_all_fields()
    timeseries_dataset = combined_datasets.build_us_timeseries_with_all_fields()

    # restricting the datasets being persisted to one county to speed up tests a bit.
    latest_nyc = latest.get_subset(None, fips=nyc_fips)
    timeseries_nyc = timeseries_dataset.get_subset(None, fips=nyc_fips)

    combined_dataset_utils.update_data_public_head(
        f"s3://{mock_s3_bucket}",
        pointer_path_dir=tmp_path,
        latest_dataset=latest_nyc,
        timeseries_dataset=timeseries_nyc,
    )

    timeseries = combined_dataset_utils.load_us_timeseries_with_all_fields(
        promotion_level=DatasetPromotion.LATEST,
        pointer_directory=tmp_path,
        dataset_download_directory=tmp_path,
    )

    latest = combined_dataset_utils.load_us_latest_with_all_fields(
        promotion_level=DatasetPromotion.LATEST,
        pointer_directory=tmp_path,
        dataset_download_directory=tmp_path,
    )

    assert latest.get_record_for_fips(nyc_fips) == latest_nyc.get_record_for_fips(nyc_fips)
