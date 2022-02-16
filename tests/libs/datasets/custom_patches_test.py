import io

from tests import test_helpers
from libs.datasets.timeseries import MultiRegionDataset
from libs.datasets.custom_patches import patch_maryland_missing_case_data


def test_patch_maryland_missing_case_data():
    ds_in = MultiRegionDataset.from_csv(
        io.StringIO(
            "location_id,date,new_cases\n"
            "iso1:us#iso2:us-md,2021-12-02,15\n"
            "iso1:us#iso2:us-md,2021-12-03,16\n"
            "iso1:us#iso2:us-md,2021-12-04,17\n"
            "iso1:us#iso2:us-md,2021-12-06,0\n"
            "iso1:us#iso2:us-md,2021-12-07,0\n"
            "iso1:us#iso2:us-md,2021-12-08,0\n"
            "iso1:us#iso2:us-md,2021-12-09,0\n"
            "iso1:us#iso2:us-md,2021-12-10,0\n"
            "iso1:us#iso2:us-md,2021-12-11,0\n"
            "iso1:us#iso2:us-md,2021-12-12,0\n"
            "iso1:us#iso2:us-md,2021-12-13,0\n"
            "iso1:us#iso2:us-md,2021-12-14,0\n"
            "iso1:us#iso2:us-md,2021-12-16,100\n"
            "iso1:us#iso2:us-md,2021-12-17,18\n"
            "iso1:us#iso2:us-md,2021-12-18,19\n"
            "iso1:us#iso2:us-ma,2021-12-12,30\n"
            "iso1:us#iso2:us-ma,2021-12-13,30\n"
        )
    )
    expected = MultiRegionDataset.from_csv(
        io.StringIO(
            "location_id,date,new_cases\n"
            "iso1:us#iso2:us-md,2021-12-02,15\n"
            "iso1:us#iso2:us-md,2021-12-03,16\n"
            "iso1:us#iso2:us-md,2021-12-04,17\n"
            "iso1:us#iso2:us-md,2021-12-06,10\n"
            "iso1:us#iso2:us-md,2021-12-07,10\n"
            "iso1:us#iso2:us-md,2021-12-08,10\n"
            "iso1:us#iso2:us-md,2021-12-09,10\n"
            "iso1:us#iso2:us-md,2021-12-10,10\n"
            "iso1:us#iso2:us-md,2021-12-11,10\n"
            "iso1:us#iso2:us-md,2021-12-12,10\n"
            "iso1:us#iso2:us-md,2021-12-13,10\n"
            "iso1:us#iso2:us-md,2021-12-14,10\n"
            "iso1:us#iso2:us-md,2021-12-16,10\n"
            "iso1:us#iso2:us-md,2021-12-17,18\n"
            "iso1:us#iso2:us-md,2021-12-18,19\n"
            "iso1:us#iso2:us-ma,2021-12-12,30\n"
            "iso1:us#iso2:us-ma,2021-12-13,30\n"
        )
    )

    result = patch_maryland_missing_case_data(ds_in)
    test_helpers.assert_dataset_like(result, expected)
