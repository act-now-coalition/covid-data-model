import pathlib

from covidactnow.datapublic.common_fields import CommonFields

from libs.datasets import combined_dataset_utils
from libs.datasets import combined_datasets
from libs.datasets.latest_values_dataset import LatestValuesDataset
from libs.pipeline import Region
from libs.qa.common_df_diff import DatasetDiff
from test.libs.datasets.timeseries_test import assert_dataset_like


def test_persist_and_load_dataset(tmp_path, nyc_fips):
    region = Region.from_fips(nyc_fips)
    dataset = combined_datasets.load_us_timeseries_dataset()
    timeseries_nyc = dataset.get_regions_subset([region]).to_timeseries()

    pointer = combined_dataset_utils.persist_dataset(timeseries_nyc, tmp_path)

    downloaded_dataset = pointer.load_dataset()
    differ_l = DatasetDiff.make(downloaded_dataset.data)
    differ_r = DatasetDiff.make(timeseries_nyc.data)
    differ_l.compare(differ_r)

    assert not len(differ_l.my_ts)


def test_update_and_load(tmp_path: pathlib.Path, nyc_fips, nyc_region):
    # restricting the datasets being persisted to one county to speed up tests a bit.
    multiregion_timeseries_nyc = combined_datasets.load_us_timeseries_dataset().get_regions_subset(
        [nyc_region]
    )
    latest_nyc = LatestValuesDataset(multiregion_timeseries_nyc.static_data_with_fips.reset_index())
    one_region_nyc = multiregion_timeseries_nyc.get_one_region(nyc_region)
    assert one_region_nyc.latest[CommonFields.POPULATION] > 1_000_000
    assert one_region_nyc.region.location_id

    combined_dataset_utils.update_data_public_head(
        tmp_path, latest_dataset=latest_nyc, timeseries_dataset=multiregion_timeseries_nyc,
    )

    timeseries_loaded = combined_datasets.load_us_timeseries_dataset(pointer_directory=tmp_path)
    one_region_loaded = timeseries_loaded.get_one_region(nyc_region)
    assert one_region_nyc.latest == one_region_loaded.latest
    assert_dataset_like(timeseries_loaded, multiregion_timeseries_nyc, drop_na_timeseries=True)
