import json
import pathlib

import structlog
from datapublic.common_fields import CommonFields
import pandas as pd

import pyseir.run
from libs.datasets import timeseries
from libs.pipeline import Region
from libs.pipelines import api_v2_pipeline
from tests import test_helpers


def test_generate_from_loaded_data_country(tmpdir):

    output_dir = pathlib.Path(tmpdir)
    model_output = pyseir.run.PyseirOutputDatasets(
        timeseries.MultiRegionDataset.new_without_timeseries(),
    )
    dataset = test_helpers.build_default_region_dataset(
        {
            CommonFields.CASES: [20, 30, 40],
            CommonFields.NEW_CASES: [10, 10, 10],
            CommonFields.ICU_BEDS: [5, 5, 5],
            CommonFields.CURRENT_ICU: [2, 2, 2],
            CommonFields.CONTACT_TRACERS_COUNT: [10, 10, 10],
            CommonFields.VACCINATIONS_COMPLETED: [300, 350, 400],
            CommonFields.VACCINATIONS_INITIATED: [400, 450, 500],
            CommonFields.VACCINES_DISTRIBUTED: [400, 450, 500],
            CommonFields.STAFFED_BEDS: [100, 200, 300],
            CommonFields.CURRENT_HOSPITALIZED: [50, 100, 150],
        },
        static={
            CommonFields.POPULATION: 1000,
            CommonFields.HSA: 202,
            CommonFields.HSA_POPULATION: 10000,
        },
        region=Region.from_iso1("us"),
    )

    api_v2_pipeline.generate_from_loaded_data(
        model_output, output_dir, dataset, structlog.get_logger(),
    )

    assert json.load((output_dir / "country" / "US.json").open())
    assert json.load((output_dir / "country" / "US.timeseries.json").open())
    assert not pd.read_csv(output_dir / "country" / "US.timeseries.csv").empty
    assert json.load((output_dir / "countries.json").open())
    assert json.load((output_dir / "countries.timeseries.json").open())
    assert not pd.read_csv(output_dir / "countries.csv").empty
