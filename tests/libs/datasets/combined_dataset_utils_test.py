import pathlib

import pytest
from covidactnow.datapublic.common_fields import CommonFields
from libs.datasets import AggregationLevel
from libs.datasets import combined_dataset_utils
from libs.datasets import combined_datasets
from libs.datasets.timeseries import MultiRegionDataset
from libs.datasets.sources.nytimes_dataset import NYTimesDataset
from libs.pipeline import Region
from libs.pipeline import RegionMask
from libs.qa.common_df_diff import DatasetDiff
from tests import test_helpers


def test_persist_and_load_dataset(tmp_path, nyc_fips):
    region = Region.from_fips(nyc_fips)
    dataset = combined_datasets.load_us_timeseries_dataset()
    timeseries_nyc = dataset.get_regions_subset([region])

    pointer = combined_dataset_utils.persist_dataset(timeseries_nyc, tmp_path)

    downloaded_dataset = MultiRegionDataset.read_from_pointer(pointer)
    differ_l = DatasetDiff.make(downloaded_dataset.timeseries)
    differ_r = DatasetDiff.make(timeseries_nyc.timeseries)
    differ_l.compare(differ_r)

    assert not len(differ_l.my_ts)


def test_update_and_load(tmp_path: pathlib.Path, nyc_fips, nyc_region):
    # restricting the datasets being persisted to one county to speed up tests a bit.
    multiregion_timeseries_nyc = combined_datasets.load_us_timeseries_dataset().get_regions_subset(
        [nyc_region]
    )
    one_region_nyc = multiregion_timeseries_nyc.get_one_region(nyc_region)
    assert one_region_nyc.latest[CommonFields.POPULATION] > 1_000_000
    assert one_region_nyc.region.location_id

    combined_dataset_utils.persist_dataset(
        multiregion_timeseries_nyc, tmp_path,
    )

    timeseries_loaded = combined_datasets.load_us_timeseries_dataset(pointer_directory=tmp_path)
    one_region_loaded = timeseries_loaded.get_one_region(nyc_region)
    assert one_region_nyc.latest == pytest.approx(one_region_loaded.latest)
    test_helpers.assert_dataset_like(
        timeseries_loaded, multiregion_timeseries_nyc, drop_na_timeseries=True
    )


def test_include_exclude_regions():

    mask = combined_datasets.DataSourceAndRegionMasks(
        NYTimesDataset,
        exclude=[
            Region.from_fips("36061"),
            RegionMask(level=AggregationLevel.COUNTY, states=["DC"]),
        ],
    )

    location_ids = mask._get_location_ids(mask.exclude)
    assert location_ids == ["iso1:us#iso2:us-ny#fips:36061", "iso1:us#iso2:us-dc#fips:11001"]
