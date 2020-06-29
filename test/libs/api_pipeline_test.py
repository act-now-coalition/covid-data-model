import datetime
import pathlib
import tempfile
import pytest
from libs.pipelines import api_pipeline
from libs.datasets import combined_datasets
from libs.enums import Intervention
from api.can_api_definition import CovidActNowAreaSummary
from api.can_api_definition import _Actuals

from pyseir import cli

#
NYC_FIPS = "36061"


@pytest.fixture(scope="module")
def pyseir_output_path():
    with tempfile.TemporaryDirectory() as tempdir:
        cli._build_all_for_states(
            states=["New York"], generate_reports=False, output_dir=tempdir, fips="36061"
        )

        yield fips, pathlib.Path(tempdir)


def test_generate_summary_for_fips():

    us_latest = combined_datasets.build_us_latest_with_all_fields()
    nyc_latest = us_latest.get_record_for_fips(NYC_FIPS)
    summary = api_pipeline.generate_area_summary_for_fips_intervention(
        NYC_FIPS, Intervention.OBSERVED_INTERVENTION, us_latest, None
    )
    expected = CovidActNowAreaSummary(
        population=nyc_latest["population"],
        stateName="New York",
        countyName="New York County",
        fips="36061",
        lat=None,
        long=None,
        actuals=_Actuals(
            population=nyc_latest["population"],
            intervention="STRONG_INTERVENTION",
            cumulativeConfirmedCases=nyc_latest["cases"],
            cumulativeDeaths=nyc_latest["deaths"],
            cumulativePositiveTests=nyc_latest["positive_tests"],
            cumulativeNegativeTests=nyc_latest["negative_tests"],
            hospitalBeds={
                # Manually calculated from capacity calculation in generate_api.py
                "capacity": 12763,
                "totalCapacity": nyc_latest["max_bed_count"],
                "currentUsageCovid": None,
                "currentUsageTotal": None,
                "typicalUsageRate": nyc_latest["all_beds_occupancy_rate"],
            },
            ICUBeds={
                "capacity": nyc_latest["icu_beds"],
                "totalCapacity": nyc_latest["icu_beds"],
                "currentUsageCovid": None,
                "currentUsageTotal": None,
                "typicalUsageRate": nyc_latest["icu_occupancy_rate"],
            },
            contactTracers=nyc_latest["contact_tracers_count"],
        ),
        lastUpdatedDate=datetime.datetime.utcnow(),
        projections=None,
    )
    assert expected == summary
