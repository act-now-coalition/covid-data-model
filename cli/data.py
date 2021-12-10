import dataclasses
import datetime
from typing import List
from typing import Mapping
from typing import Optional
import logging
import pathlib
import json
import structlog

import click
from covidactnow.datapublic.common_fields import CommonFields
from covidactnow.datapublic.common_fields import FieldName

from libs import google_sheet_helpers
from libs import pipeline
from libs.datasets import combined_dataset_utils
from libs.datasets import nytimes_anomalies
from libs.datasets import custom_aggregations
from libs.datasets import manual_filter
from libs.datasets import statistical_areas
from libs.datasets.combined_datasets import (
    ALL_TIMESERIES_FEATURE_DEFINITION,
    ALL_FIELDS_FEATURE_DEFINITION,
)
from libs.datasets import timeseries
from libs.datasets import outlier_detection
from libs.datasets import dataset_utils
from libs.datasets import combined_datasets
from libs.datasets import new_cases_and_deaths
from libs.datasets import vaccine_backfills
from libs.datasets.dataset_utils import DATA_DIRECTORY
from libs.datasets import tail_filter
from libs.datasets.sources import zeros_filter
from libs.pipeline import Region
from libs.pipeline import RegionMask
from libs.us_state_abbrev import ABBREV_US_UNKNOWN_COUNTY_FIPS


TailFilter = tail_filter.TailFilter


CUMULATIVE_FIELDS_TO_FILTER = [
    CommonFields.CASES,
    CommonFields.DEATHS,
    CommonFields.POSITIVE_TESTS,
    CommonFields.NEGATIVE_TESTS,
    CommonFields.TOTAL_TESTS,
    CommonFields.POSITIVE_TESTS_VIRAL,
    CommonFields.POSITIVE_CASES_VIRAL,
    CommonFields.TOTAL_TESTS_VIRAL,
    CommonFields.TOTAL_TESTS_PEOPLE_VIRAL,
    CommonFields.TOTAL_TEST_ENCOUNTERS_VIRAL,
]

PROD_BUCKET = "data.covidactnow.org"

# By default require 0.95 of populations from regions to include a data point in aggregate.
DEFAULT_REPORTING_RATIO = 0.95

_logger = logging.getLogger(__name__)


REGION_OVERRIDES_JSON = DATA_DIRECTORY / "region-overrides.json"


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
    help="Disable to skip loading datasets from covid-data-public and instead re-use data from combined-raw.pkl.gz (much faster)",
    default=True,
)
@click.option("--state", type=str, help="For testing, a two letter state abbr")
@click.option("--fips", type=str, help="For testing, a 5 digit county fips")
def update(
    aggregate_to_country: bool,
    print_stats: bool,
    refresh_datasets: bool,
    state: Optional[str],
    fips: Optional[str],
):
    _logger.info("1: update()")
    """Updates latest and timeseries datasets to the current checked out covid data public commit"""
    path_prefix = dataset_utils.DATA_DIRECTORY.relative_to(dataset_utils.REPO_ROOT)

    if refresh_datasets:
        _logger.info("2: refresh_datasets")
        timeseries_field_datasets = load_datasets_by_field(
            ALL_TIMESERIES_FEATURE_DEFINITION, state=state, fips=fips
        )
        static_field_datasets = load_datasets_by_field(
            ALL_FIELDS_FEATURE_DEFINITION, state=state, fips=fips
        )

        multiregion_dataset = timeseries.combined_datasets(
            timeseries_field_datasets, static_field_datasets
        )
        _logger.info("Finished combining datasets")
        # HACK(michael): Remove demographic data.
        multiregion_dataset.timeseries_bucketed = multiregion_dataset.timeseries_bucketed.loc[
            multiregion_dataset.timeseries_bucketed.index.get_level_values("demographic_bucket")
            == "all"
        ]
        multiregion_dataset.to_compressed_pickle(dataset_utils.COMBINED_RAW_PICKLE_GZ_PATH)
        if print_stats:
            multiregion_dataset.print_stats("combined")
    else:
        multiregion_dataset = timeseries.MultiRegionDataset.from_compressed_pickle(
            dataset_utils.COMBINED_RAW_PICKLE_GZ_PATH
        )

    # Apply manual overrides (currently only removing timeseries) before aggregation so we don't
    # need to remove CBSAs because they don't exist yet.
    aggregator = statistical_areas.CountyToCBSAAggregator.from_local_public_data()
    region_overrides_config = manual_filter.transform_region_overrides(
        json.load(open(REGION_OVERRIDES_JSON)), aggregator.cbsa_to_counties_region_map
    )
    before_manual_filter = multiregion_dataset
    multiregion_dataset = manual_filter.run(multiregion_dataset, region_overrides_config)
    manual_filter_touched = manual_filter.touched_subset(before_manual_filter, multiregion_dataset)
    manual_filter_touched.write_to_wide_dates_csv(
        dataset_utils.MANUAL_FILTER_REMOVED_WIDE_DATES_CSV_PATH,
        dataset_utils.MANUAL_FILTER_REMOVED_STATIC_CSV_PATH,
    )
    if print_stats:
        multiregion_dataset.print_stats("manual filter")

    multiregion_dataset = timeseries.drop_observations(
        multiregion_dataset, after=datetime.datetime.utcnow().date()
    )

    multiregion_dataset = outlier_detection.drop_tail_positivity_outliers(multiregion_dataset)
    if print_stats:
        multiregion_dataset.print_stats("drop_tail")
    # Filter for stalled cumulative values before deriving NEW_CASES from CASES.
    _, multiregion_dataset = TailFilter.run(multiregion_dataset, CUMULATIVE_FIELDS_TO_FILTER)
    if print_stats:
        multiregion_dataset.print_stats("TailFilter")
    multiregion_dataset = zeros_filter.drop_all_zero_timeseries(
        multiregion_dataset,
        [
            CommonFields.VACCINES_DISTRIBUTED,
            CommonFields.VACCINES_ADMINISTERED,
            CommonFields.VACCINATIONS_COMPLETED,
            CommonFields.VACCINATIONS_INITIATED,
        ],
    )
    if print_stats:
        multiregion_dataset.print_stats("zeros_filter")

    multiregion_dataset = vaccine_backfills.estimate_initiated_from_state_ratio(multiregion_dataset)
    if print_stats:
        multiregion_dataset.print_stats("estimate_initiated_from_state_ratio")

    multiregion_dataset = new_cases_and_deaths.add_new_cases(multiregion_dataset)
    multiregion_dataset = new_cases_and_deaths.add_new_deaths(multiregion_dataset)
    if print_stats:
        multiregion_dataset.print_stats("new_cases_and_deaths")

    multiregion_dataset = nytimes_anomalies.filter_by_nyt_anomalies(multiregion_dataset)
    if print_stats:
        multiregion_dataset.print_stats("nytimes_anomalies")

    multiregion_dataset = outlier_detection.drop_new_case_outliers(multiregion_dataset)
    multiregion_dataset = outlier_detection.drop_new_deaths_outliers(multiregion_dataset)
    if print_stats:
        multiregion_dataset.print_stats("outlier_detection")

    multiregion_dataset = timeseries.drop_regions_without_population(
        multiregion_dataset, KNOWN_LOCATION_ID_WITHOUT_POPULATION, structlog.get_logger()
    )
    if print_stats:
        multiregion_dataset.print_stats("drop_regions_without_population")

    multiregion_dataset = custom_aggregations.aggregate_puerto_rico_from_counties(
        multiregion_dataset
    )
    if print_stats:
        multiregion_dataset.print_stats("aggregate_puerto_rico_from_counties")
    multiregion_dataset = custom_aggregations.aggregate_to_new_york_city(multiregion_dataset)
    if print_stats:
        multiregion_dataset.print_stats("aggregate_to_new_york_city")
    multiregion_dataset = custom_aggregations.replace_dc_county_with_state_data(multiregion_dataset)
    if print_stats:
        multiregion_dataset.print_stats("replace_dc_county_with_state_data")

    cbsa_dataset = aggregator.aggregate(
        multiregion_dataset, reporting_ratio_required_to_aggregate=DEFAULT_REPORTING_RATIO
    )
    multiregion_dataset = multiregion_dataset.append_regions(cbsa_dataset)
    if print_stats:
        multiregion_dataset.print_stats("CountyToCBSAAggregator")

    # TODO(tom): Add a clean way to store intermediate values instead of commenting out code like
    #  this:
    # multiregion_dataset.write_to_wide_dates_csv(
    #     pathlib.Path("data/pre-agg-wide-dates.csv"), pathlib.Path("data/pre-agg-static.csv")
    # )
    if aggregate_to_country:
        multiregion_dataset = custom_aggregations.aggregate_to_country(
            multiregion_dataset, reporting_ratio_required_to_aggregate=DEFAULT_REPORTING_RATIO
        )
        if print_stats:
            multiregion_dataset.print_stats("aggregate_to_country")

    combined_dataset_utils.persist_dataset(multiregion_dataset, path_prefix)
    if print_stats:
        multiregion_dataset.print_stats("persist")


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


KNOWN_LOCATION_ID_WITHOUT_POPULATION = [
    # Territories other than PR
    "iso1:us#iso2:us-vi",
    "iso1:us#iso2:us-as",
    "iso1:us#iso2:us-gu",
    # Subregion of AS
    "iso1:us#iso2:us-vi#fips:78030",
    "iso1:us#iso2:us-vi#fips:78020",
    "iso1:us#iso2:us-vi#fips:78010",
    # Retired FIPS
    "iso1:us#iso2:us-sd#fips:46113",
    "iso1:us#iso2:us-va#fips:51515",
    # All the unknown county FIPS
    *[pipeline.fips_to_location_id(f) for f in ABBREV_US_UNKNOWN_COUNTY_FIPS.values()],
]


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
        test_subset = dataclasses.replace(
            test_subset, timeseries_bucketed=test_subset.timeseries_bucketed.loc[date_range_mask]
        )
    test_subset.write_to_wide_dates_csv(
        dataset_utils.TEST_COMBINED_WIDE_DATES_CSV_PATH, dataset_utils.TEST_COMBINED_STATIC_CSV_PATH
    )


def load_datasets_by_field(
    feature_definition_config: combined_datasets.FeatureDataSourceMap, *, state, fips
) -> Mapping[FieldName, List[timeseries.MultiRegionDataset]]:
    def _load_dataset(data_source_cls) -> timeseries.MultiRegionDataset:
        try:
            dataset = data_source_cls.make_dataset()
            if state or fips:
                dataset = dataset.get_subset(state=state, fips=fips)
            return dataset
        except Exception:
            raise ValueError(f"Problem with {data_source_cls}")

    feature_definition = {
        # Put the highest priority first, as expected by timeseries.combined_datasets.
        # TODO(tom): reverse the hard-coded FeatureDataSourceMap and remove the reversed call.
        field_name: list(reversed(list(_load_dataset(cls) for cls in classes)))
        for field_name, classes in feature_definition_config.items()
        if classes
    }
    return feature_definition
