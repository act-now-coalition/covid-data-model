import pathlib

from covidactnow.datapublic.common_fields import CommonFields

from libs.datasets import combined_dataset_utils
from libs.datasets import combined_datasets
from libs.datasets.timeseries import MultiRegionTimeseriesDataset
from libs.datasets.timeseries import TimeseriesDataset
from libs.pipeline import Region
from libs.qa.common_df_diff import DatasetDiff


def test_persist_and_load_dataset(tmp_path, nyc_fips):
    region = Region.from_fips(nyc_fips)
    dataset = combined_datasets.load_us_timeseries_dataset()
    timeseries_nyc = TimeseriesDataset(dataset.get_one_region(region).data)

    pointer = combined_dataset_utils.persist_dataset(timeseries_nyc, tmp_path)

    downloaded_dataset = pointer.load_dataset()
    differ_l = DatasetDiff.make(downloaded_dataset.data)
    differ_r = DatasetDiff.make(timeseries_nyc.data)
    differ_l.compare(differ_r)

    assert not len(differ_l.my_ts)


def test_update_and_load(tmp_path: pathlib.Path, nyc_fips, nyc_region):
    us_combined_df = combined_datasets.load_us_timeseries_dataset().combined_df

    # restricting the datasets being persisted to one county to speed up tests a bit.
    nyc_combined_df = us_combined_df.loc[us_combined_df[CommonFields.FIPS] == nyc_fips, :]
    multiregion_timeseries_nyc = MultiRegionTimeseriesDataset.from_combined_dataframe(
        nyc_combined_df
    )
    timeseries_nyc = multiregion_timeseries_nyc.to_timeseries()
    latest_nyc = timeseries_nyc.latest_values_object()

    combined_dataset_utils.update_data_public_head(
        tmp_path, latest_dataset=latest_nyc, timeseries_dataset=multiregion_timeseries_nyc,
    )

    timeseries = combined_datasets.load_us_timeseries_dataset(pointer_directory=tmp_path)

    latest = combined_datasets.load_us_latest_dataset(pointer_directory=tmp_path)

    assert latest.get_record_for_fips(nyc_fips) == latest_nyc.get_record_for_fips(nyc_fips)
