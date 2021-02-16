import pytest
from covidactnow.datapublic.common_fields import CommonFields

from api import can_api_v2_definition
from api.can_api_v2_definition import AnomalyAnnotation
from api.can_api_v2_definition import FieldSource
from libs import build_api_v2
from libs.datasets.dataset_utils import GEO_DATA_COLUMNS
from libs.datasets.dataset_utils import TIMESERIES_INDEX_FIELDS
from libs.datasets.taglib import UrlStr
from libs.metrics import test_positivity
from libs.datasets import timeseries
from libs.pipeline import Region
from libs.pipelines import api_v2_pipeline
from libs.datasets import combined_datasets
from libs.datasets import AggregationLevel
import pandas as pd
import structlog

from tests import test_helpers
from tests.test_helpers import TimeseriesLiteral


@pytest.fixture
def nyc_regional_input(nyc_region, rt_dataset, icu_dataset):
    us_dataset = combined_datasets.load_us_timeseries_dataset()
    # Not using test_positivity because currently we don't have any data for counties
    return api_v2_pipeline.RegionalInput.from_region_and_model_output(
        nyc_region, us_dataset, rt_dataset, icu_dataset
    )


@pytest.fixture
def il_regional_input(rt_dataset, icu_dataset):
    region = Region.from_state("IL")
    regional_data = combined_datasets.load_us_timeseries_dataset().get_regions_subset([region])
    regional_data = test_positivity.run_and_maybe_join_columns(
        regional_data, structlog.get_logger()
    )

    return api_v2_pipeline.RegionalInput.from_region_and_model_output(
        region, regional_data, rt_dataset, icu_dataset
    )


@pytest.fixture
def il_regional_input_empty_test_positivity_column(rt_dataset, icu_dataset):
    region = Region.from_state("IL")
    regional_data = combined_datasets.load_us_timeseries_dataset().get_regions_subset([region])
    empty_test_positivity = timeseries.MultiRegionDataset.from_geodata_timeseries_df(
        pd.DataFrame(
            [], columns=[CommonFields.LOCATION_ID, CommonFields.DATE, CommonFields.TEST_POSITIVITY]
        )
    )

    regional_data = regional_data.drop_column_if_present(CommonFields.TEST_POSITIVITY).join_columns(
        empty_test_positivity
    )
    return api_v2_pipeline.RegionalInput.from_region_and_model_output(
        region, regional_data, rt_dataset, icu_dataset
    )


def test_build_timeseries_and_summary_outputs(nyc_regional_input):
    timeseries = api_v2_pipeline.build_timeseries_for_region(nyc_regional_input)
    assert timeseries
    assert timeseries.riskLevels.testPositivityRatio
    assert timeseries.metrics.testPositivityRatioDetails.source


def test_build_timeseries_and_summary_outputs_for_il_state(il_regional_input):
    timeseries = api_v2_pipeline.build_timeseries_for_region(il_regional_input)
    assert timeseries


def test_build_timeseries_and_summary_outputs_for_il_state_with_empty_test_postivity_columnn(
    il_regional_input_empty_test_positivity_column,
):
    timeseries = api_v2_pipeline.build_timeseries_for_region(
        il_regional_input_empty_test_positivity_column
    )
    assert timeseries


def test_build_api_output_for_intervention(nyc_regional_input, tmp_path):
    county_output = tmp_path
    all_timeseries_api = api_v2_pipeline.run_on_regions([nyc_regional_input])

    api_v2_pipeline.deploy_single_level(all_timeseries_api, AggregationLevel.COUNTY, county_output)
    expected_outputs = [
        "counties.timeseries.json",
        "counties.csv",
        "counties.timeseries.csv",
        "counties.json",
        "county/36061.json",
        "county/36061.timeseries.json",
        "county/NY.timeseries.json",
        "county/NY.timeseries.csv",
        "county/NY.json",
        "county/NY.csv",
    ]

    output_paths = [
        str(path.relative_to(tmp_path)) for path in tmp_path.glob("**/*") if not path.is_dir()
    ]
    assert set(output_paths) == set(expected_outputs)


def test_output_no_timeseries_rows(nyc_regional_input, tmp_path):

    # Creating a new regional input with an empty timeseries dataset
    timeseries = nyc_regional_input.timeseries
    one_region = combined_datasets.load_us_timeseries_dataset().get_one_region(
        nyc_regional_input.region
    )
    regional_input = api_v2_pipeline.RegionalInput(
        nyc_regional_input.region, one_region, None, None
    )
    assert not regional_input.timeseries.empty

    all_timeseries_api = api_v2_pipeline.run_on_regions([regional_input])

    assert all_timeseries_api


def test_annotation(rt_dataset, icu_dataset):
    region = Region.from_state("IL")
    tag = test_helpers.make_tag(date="2020-04-01", original_observation=10.0)
    death_url = UrlStr("http://can.com/death_source")
    cases_urls = [UrlStr("http://can.com/one"), UrlStr("http://can.com/two")]
    new_cases_url = UrlStr("http://can.com/new_cases")

    ds = test_helpers.build_default_region_dataset(
        {
            CommonFields.CASES: TimeseriesLiteral(
                [100, 200, 300], provenance="NYTimes", source_url=cases_urls
            ),
            # NEW_CASES has only source_url set, to make sure that an annotation is still output.
            CommonFields.NEW_CASES: TimeseriesLiteral([100, 100, 100], source_url=new_cases_url),
            CommonFields.CONTACT_TRACERS_COUNT: [10] * 3,
            CommonFields.ICU_BEDS: TimeseriesLiteral([20, 20, 20], provenance="NotFound"),
            CommonFields.CURRENT_ICU: [5, 5, 5],
            CommonFields.DEATHS: TimeseriesLiteral(
                [2, 3, 2], annotation=[tag], source_url=death_url
            ),
        },
        region=region,
        static={
            CommonFields.POPULATION: 100_000,
            CommonFields.STATE: "IL",
            CommonFields.CAN_LOCATION_PAGE_URL: "http://covidactnow.org/foo/bar",
        },
    )
    regional_input = api_v2_pipeline.RegionalInput.from_region_and_model_output(
        region, ds, rt_dataset, icu_dataset
    )

    with structlog.testing.capture_logs() as logs:
        timeseries_for_region = api_v2_pipeline.build_timeseries_for_region(regional_input)

    assert logs == [
        {
            "location_id": region.location_id,
            "field_name": CommonFields.CASES,
            "event": build_api_v2.METRIC_MULTIPLE_SOURCE_URLS_MESSAGE,
            "log_level": "warning",
            "urls": cases_urls,
        },
        {
            "location_id": region.location_id,
            "field_name": CommonFields.ICU_BEDS,
            "provenance": "NotFound",
            "event": build_api_v2.METRIC_SOURCES_NOT_FOUND_MESSAGE,
            "log_level": "info",
        },
    ]
    assert timeseries_for_region.annotations.icuBeds.sources == [FieldSource.OTHER]
    assert timeseries_for_region.annotations.icuBeds.anomalies == []

    assert timeseries_for_region.annotations.cases.sources == [FieldSource.NYTimes]
    assert timeseries_for_region.annotations.cases.anomalies == []
    # _build_metric_annotations picks one source_url at random.
    assert timeseries_for_region.annotations.cases.source_url in cases_urls

    assert timeseries_for_region.annotations.deaths.sources == []
    assert timeseries_for_region.annotations.deaths.source_url == death_url
    assert timeseries_for_region.annotations.deaths.anomalies == [
        AnomalyAnnotation(date="2020-04-01", original_observation=10.0, type=tag.type)
    ]

    assert timeseries_for_region.annotations.newCases.source_url == new_cases_url

    assert timeseries_for_region.annotations.contactTracers is None


def test_annotation_all_fields_copied(rt_dataset, icu_dataset):
    region = Region.from_state("IL")
    # Create a dataset with bogus data for every CommonFields, excluding a few that are not
    # expected to have timeseries values.
    fields_excluded = {*TIMESERIES_INDEX_FIELDS, *GEO_DATA_COLUMNS, CommonFields.LOCATION_ID}
    ds = test_helpers.build_default_region_dataset(
        {
            field: TimeseriesLiteral([100, 200, 300], provenance="NYTimes")
            for field in CommonFields
            if field not in fields_excluded
        },
        region=region,
        static={
            CommonFields.POPULATION: 100_000,
            CommonFields.STATE: "IL",
            CommonFields.CAN_LOCATION_PAGE_URL: "http://covidactnow.org/foo/bar",
        },
    )
    regional_input = api_v2_pipeline.RegionalInput.from_region_and_model_output(
        region, ds, rt_dataset, icu_dataset
    )

    timeseries_for_region = api_v2_pipeline.build_timeseries_for_region(regional_input)

    # Check that build_annotations set every field in Annotations.
    for field_name in can_api_v2_definition.Annotations.__fields__.keys():
        assert getattr(timeseries_for_region.annotations, field_name), f"{field_name} not set"
