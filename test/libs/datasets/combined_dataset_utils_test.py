import pathlib

from covidactnow.datapublic.common_fields import CommonFields

from libs.datasets import combined_dataset_utils
from libs.datasets import combined_datasets
from libs.datasets.latest_values_dataset import LatestValuesDataset
from libs.datasets.timeseries import TimeseriesDataset
from libs.pipeline import Region
from libs.qa.common_df_diff import DatasetDiff
from test.libs.datasets.timeseries_test import assert_combined_like


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
    latest_nyc = LatestValuesDataset(multiregion_timeseries_nyc.latest_data_with_fips.reset_index())
    latest_nyc_record = latest_nyc.get_record_for_fips(nyc_fips)
    assert latest_nyc_record[CommonFields.POPULATION] > 1_000_000
    assert latest_nyc_record[CommonFields.LOCATION_ID]

    combined_dataset_utils.update_data_public_head(
        tmp_path, latest_dataset=latest_nyc, timeseries_dataset=multiregion_timeseries_nyc,
    )

    timeseries_loaded = combined_datasets.load_us_timeseries_dataset(pointer_directory=tmp_path)
    latest_loaded = combined_datasets.load_us_latest_dataset(pointer_directory=tmp_path)
    assert latest_loaded.get_record_for_fips(nyc_fips) == latest_nyc_record
    assert_combined_like(timeseries_loaded, multiregion_timeseries_nyc, drop_na_timeseries=True)
