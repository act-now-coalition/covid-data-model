import click
import logging
import pathlib
import structlog
import dataclasses
from typing import Optional, List

from libs import google_sheet_helpers
from libs.pipeline import Region, RegionMask
from datapublic.common_fields import CommonFields
from libs.datasets.timeseries_orchestrator import (
    MultiRegionOrchestrator,
    KNOWN_LOCATION_ID_WITHOUT_POPULATION,
)
from libs.datasets.dataset_utils import (
    CUMULATIVE_FIELDS_TO_FILTER,
    DEFAULT_REPORTING_RATIO,
)
from libs.datasets import (
    combined_dataset_utils,
    custom_aggregations,
    statistical_areas,
    timeseries,
    dataset_utils,
    combined_datasets,
    tail_filter,
)

DATA_PATH_PREFIX = dataset_utils.DATA_DIRECTORY.relative_to(dataset_utils.REPO_ROOT)

TailFilter = tail_filter.TailFilter

_logger = logging.getLogger(__name__)


@click.group("data")
def main():
    pass


@main.command()
@click.option(
    "--aggregate-to-country/--no-aggregate-to-country",
    is_flag=True,
    help="Aggregate states to one USA country region",
    default=True,
)
@click.option(
    "--print-stats/--no-print-stats",
    is_flag=True,
    help="Print summary stats at several places in the pipeline. Producing these takes extra time.",
    default=True,
)
@click.option(
    "--refresh-datasets/--no-refresh-datasets",
    is_flag=True,
    help="Disable to skip loading datasets and instead re-use data from combined-raw.pkl.gz (much faster)",
    default=True,
)
@click.option(
    "--states", "-s", type=str, multiple=True, help="For testing, two letter state abbrev's"
)
def update(
    aggregate_to_country: bool,
    print_stats: bool,
    refresh_datasets: bool,
    states: Optional[List[str]],
):
    dataset = MultiRegionOrchestrator.from_bulk_mrds(
        states=states, refresh_datasets=refresh_datasets, print_stats=print_stats
    ).build_and_combine_regions(aggregate_to_country=aggregate_to_country)
    _logger.info("Writing multiregion_dataset to disk...")
    combined_dataset_utils.persist_dataset(dataset, DATA_PATH_PREFIX)
    _logger.info("Finished writing multiregion_dataset!")


@main.command()
@click.option(
    "--print-stats/--no-print-stats",
    is_flag=True,
    help="Print summary stats at several places in the pipeline. Producing these takes extra time.",
    default=True,
)
@click.option(
    "--states", "-s", type=str, multiple=True, help="Two letter state abbrev's of states to rerun."
)
def update_and_replace_states(print_stats: bool, states: List[str]):
    # Not refreshing datasets to ensure all data comes from the same parquet file.
    dataset = MultiRegionOrchestrator.from_bulk_mrds(
        states=states, print_stats=print_stats, refresh_datasets=False
    ).update_and_replace_states()
    _logger.info("Writing multiregion_dataset to disk...")
    combined_dataset_utils.persist_dataset(dataset, DATA_PATH_PREFIX)
    _logger.info("Finished writing multiregion_dataset!")


@main.command()
@click.argument("output_path", type=pathlib.Path)
def aggregate_cbsa(output_path: pathlib.Path):
    us_timeseries = combined_datasets.load_us_timeseries_dataset()
    aggregator = statistical_areas.CountyToCBSAAggregator.from_local_public_data()
    cbsa_dataset = aggregator.aggregate(us_timeseries)
    cbsa_dataset.to_csv(output_path)


@main.command(
    short_help="Copy data from a gzipped pickle to wide dates CSV",
    help="Copy data from a gzipped pickle with path ending ....pkl.gz to a CSV with \n"
    "path argument ending ...-wide-dates.csv and ...-static.csv (derived from \n"
    "argument).\n"
    "Example: `data pickle-to-csv data/test.pkl.gz data/test-wide-dates.csv`",
)
@click.argument("pkl_gz_input", type=pathlib.Path)
@click.argument("wide_dates_csv_output", type=pathlib.Path)
def pickle_to_csv(pkl_gz_input: pathlib.Path, wide_dates_csv_output):
    assert wide_dates_csv_output.name.endswith("-wide-dates.csv")
    static_csv_output = pathlib.Path(
        str(wide_dates_csv_output).replace("-wide-dates.csv", "-static.csv")
    )
    dataset = timeseries.MultiRegionDataset.from_compressed_pickle(pkl_gz_input)
    dataset.write_to_wide_dates_csv(wide_dates_csv_output, static_csv_output)


@main.command(
    help="Uncomment code that writes the `pre-agg` intermediate result in `data update` "
    "then use this command to test state to country aggregation."
)
def aggregate_states_to_country():
    dataset = timeseries.MultiRegionDataset.from_wide_dates_csv(
        pathlib.Path("data/pre-agg-wide-dates.csv")
    ).add_static_csv_file(pathlib.Path("data/pre-agg-static.csv"))
    dataset = custom_aggregations.aggregate_to_country(
        dataset, reporting_ratio_required_to_aggregate=DEFAULT_REPORTING_RATIO
    )
    dataset.write_to_wide_dates_csv(
        pathlib.Path("data/post-agg-wide-dates.csv"), pathlib.Path("data/post-agg-static.csv")
    )


@main.command()
@click.argument("output_path", type=pathlib.Path)
def run_population_filter(output_path: pathlib.Path):
    us_timeseries = combined_datasets.load_us_timeseries_dataset()
    log = structlog.get_logger()
    log.info("starting filter")
    ts_out = timeseries.drop_regions_without_population(
        us_timeseries, KNOWN_LOCATION_ID_WITHOUT_POPULATION, log
    )
    ts_out.to_csv(output_path)


@main.command()
@click.argument("output_path", type=pathlib.Path)
def write_combined_datasets(output_path: pathlib.Path):
    log = structlog.get_logger()
    log.info("Loading")
    us_timeseries = combined_datasets.load_us_timeseries_dataset()
    log.info("Writing")
    us_timeseries.to_compressed_pickle(output_path)
    log.info("Done")


@main.command()
@click.argument("output_path", type=pathlib.Path)
def run_bad_tails_filter(output_path: pathlib.Path):
    us_dataset = combined_datasets.load_us_timeseries_dataset()
    log = structlog.get_logger()
    log.info("Starting filter")
    _, dataset_out = TailFilter.run(us_dataset, CUMULATIVE_FIELDS_TO_FILTER)
    log.info("Writing output")
    dataset_out.timeseries_rows().to_csv(output_path, index=True, float_format="%.05g")


@main.command()
@click.option("--name", envvar="DATA_AVAILABILITY_SHEET_NAME", default="Data Availability - Dev")
@click.option("--share-email")
def update_availability_report(name: str, share_email: Optional[str]):
    from libs.qa import data_availability

    sheet = google_sheet_helpers.open_or_create_spreadsheet(name, share_email=share_email)
    info_worksheet = google_sheet_helpers.update_info_sheet(sheet)
    data_sources_by_source_name = data_availability.load_all_latest_sources()

    for name, dataset in data_sources_by_source_name.items():
        _logger.info(f"Updating {name}")
        report = data_availability.build_data_availability_report(dataset)
        data_availability.update_multi_field_availability_report(
            sheet, report, name, columns_to_drop=["source", "fips"]
        )

    # Reorder sheets with combined data first and metadata last
    COLUMN_ORDER_OVERRIDE = {data_availability.COMBINED_DATA_KEY: -5, info_worksheet.title: 5}
    worksheets = sheet.worksheets()
    worksheets = sorted(worksheets, key=lambda x: (COLUMN_ORDER_OVERRIDE.get(x.title, 0), x.title))
    sheet.reorder_worksheets(worksheets)

    _logger.info("Finished updating data availability report")


@main.command(
    help="Regenerate the test combined data. The options can be used to produce data for "
    "testing or experimenting with particular subsets of the entire dataset. Use "
    "the default options when producing test data to merge into the main branch."
)
@click.option(
    "--truncate-dates/--no-truncate-dates",
    is_flag=True,
    help="Keep a subset of all dates to reduce the test data size",
    default=True,
    show_default=True,
)
@click.option(
    "--state",
    multiple=True,
    help="State to include in test data. Repeat for multiple states: --state TX --state WI.",
    default=["NY", "CA", "IL"],
    show_default=True,
)
def update_test_combined_data(truncate_dates: bool, state: List[str]):
    us_dataset = combined_datasets.load_us_timeseries_dataset()
    # Keep only a small subset of the regions so we have enough to exercise our code in tests.
    test_subset = us_dataset.get_regions_subset(
        [
            RegionMask(states=[s.strip() for s in state]),
            Region.from_fips("48201"),
            Region.from_fips("48301"),
            Region.from_fips("20161"),
            Region.from_state("TX"),
            Region.from_state("KS"),
        ]
    )
    if truncate_dates:
        dates = test_subset.timeseries_bucketed.index.get_level_values(CommonFields.DATE)
        date_range_mask = (dates >= "2021-01-01") & (dates < "2021-04-01")
        new_timeseries_bucketed = test_subset.timeseries_bucketed.loc[date_range_mask]

        # HACK: CDC_COMMUNITY_LEVEL data was added much after 2021-04-01, so the column will be all NaNs. Instead set to
        # 0 (LOW) so that the column doesn't end up getting dropped.
        buckets = new_timeseries_bucketed.index.get_level_values("demographic_bucket")
        new_timeseries_bucketed.loc[buckets == "all", CommonFields.CDC_COMMUNITY_LEVEL] = 0

        test_subset = dataclasses.replace(test_subset, timeseries_bucketed=new_timeseries_bucketed)

    test_subset.write_to_wide_dates_csv(
        dataset_utils.TEST_COMBINED_WIDE_DATES_CSV_PATH, dataset_utils.TEST_COMBINED_STATIC_CSV_PATH
    )
