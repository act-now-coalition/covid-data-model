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


def test_build_and_persist_dataset(mock_s3_bucket: str, tmp_path):
    s3_prefix = f"s3://{mock_s3_bucket}"
    dataset = combined_datasets.build_us_timeseries_with_all_fields()
    pointer = combined_dataset_utils.persist_dataset(dataset, s3_prefix, DatasetPromotion.LATEST)

    downloaded_dataset = pointer.load(download_directory=tmp_path)
    differ_l = DatasetDiff.make(downloaded_dataset.data)
    differ_r = DatasetDiff.make(dataset.data)
    differ_l.compare(differ_r)

    assert not len(differ_l.my_ts)
