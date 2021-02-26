import datetime
import io
import pathlib
import dataclasses
import pickle

import pytest
import pandas as pd
import structlog

from covidactnow.datapublic.common_fields import CommonFields
from covidactnow.datapublic.common_fields import FieldName
from covidactnow.datapublic.common_fields import PdFields

from covidactnow.datapublic.common_test_helpers import to_dict

from libs import github_utils
from libs.datasets import AggregationLevel
from libs.datasets import combined_datasets
from libs.datasets import dataset_pointer
from libs.datasets import taglib

from libs.datasets import timeseries
from libs.datasets.taglib import TagType
from libs.datasets.taglib import UrlStr
from libs.pipeline import Region
from tests import test_helpers
from tests.dataset_utils_test import read_csv_and_index_fips
from tests.dataset_utils_test import read_csv_and_index_fips_date
from tests.test_helpers import TimeseriesLiteral
from libs.datasets import vaccine_backfills


@pytest.mark.parametrize(
    "initiated_values,initiated_expected", [([50, None], [50, None]), ([None, None], [50, 150])]
)
def test_backfill_vaccine_initiated(initiated_values, initiated_expected):
    ny_region = Region.from_state("NY")
    az_region = Region.from_state("AZ")

    # Initiated has a hole, but since they're reporting some data we don't
    # want to fill
    ny_metrics = {
        CommonFields.VACCINES_ADMINISTERED: [100, 200],
        CommonFields.VACCINATIONS_INITIATED: initiated_values,
        CommonFields.VACCINATIONS_COMPLETED: [50, 50],
    }
    az_metrics = {
        CommonFields.VACCINES_ADMINISTERED: [100, 200],
        CommonFields.VACCINATIONS_COMPLETED: [30, 40],
        CommonFields.VACCINATIONS_INITIATED: [70, 160],
    }
    metrics = {ny_region: ny_metrics, az_region: az_metrics}
    dataset = test_helpers.build_dataset(metrics)
    result = vaccine_backfills.backfill_vaccination_initiated(dataset)
    expected_ny = {
        CommonFields.VACCINES_ADMINISTERED: [100, 200],
        CommonFields.VACCINATIONS_COMPLETED: [50, 50],
        CommonFields.VACCINATIONS_INITIATED: initiated_expected,
    }
    expected_metrics = {ny_region: expected_ny, az_region: az_metrics}
    expected_dataset = test_helpers.build_dataset(expected_metrics)

    test_helpers.assert_dataset_like(result, expected_dataset)


def test_derive_vaccine_pct():
    region_tx = Region.from_state("TX")
    region_sf = Region.from_fips("06075")
    # TX has metrics that will be transformed to a percent of the population. Include some other
    # data to make sure it is not dropped.
    tx_timeseries_in = {
        CommonFields.VACCINATIONS_INITIATED: TimeseriesLiteral(
            [1_000, 2_000],
            annotation=[test_helpers.make_tag(date="2020-04-01")],
            provenance=["prov1"],
        ),
        CommonFields.VACCINATIONS_COMPLETED: [None, 1_000],
        CommonFields.CASES: TimeseriesLiteral([1, 2], provenance=["caseprov"]),
    }
    # SF does not have any vaccination metrics that can be transformed to a percentage so for it
    # the input and output are the same.
    sf_timeseries = {
        CommonFields.VACCINATIONS_COMPLETED_PCT: [0.1, 1],
    }
    static_data_map = {
        region_tx: {CommonFields.POPULATION: 100_000},
        region_sf: {CommonFields.POPULATION: 10_000},
    }

    ds_in = test_helpers.build_dataset(
        {region_tx: tx_timeseries_in, region_sf: sf_timeseries},
        static_by_region_then_field_name=static_data_map,
    )

    ds_out = vaccine_backfills.derive_vaccine_pct(ds_in)

    ds_expected = test_helpers.build_dataset(
        {
            region_tx: {
                **tx_timeseries_in,
                CommonFields.VACCINATIONS_INITIATED_PCT: [1, 2],
                CommonFields.VACCINATIONS_COMPLETED_PCT: [None, 1],
            },
            region_sf: sf_timeseries,
        },
        static_by_region_then_field_name=static_data_map,
    )

    test_helpers.assert_dataset_like(ds_out, ds_expected)
