import datetime
import pathlib
import tempfile
import pytest
from libs.functions import generate_api
from libs.pipelines import api_pipeline
from libs.datasets import combined_datasets
from libs.datasets.sources.can_pyseir_location_output import CANPyseirLocationOutput
from libs.enums import Intervention
from api.can_api_definition import CovidActNowAreaSummary
from api.can_api_definition import CovidActNowAreaTimeseries
from api.can_api_definition import _Actuals
from api.can_api_definition import _Projections
from api.can_api_definition import _ResourceUsageProjection


@pytest.mark.parametrize("include_projections", [True, False])
def test_build_summary_for_fips(include_projections, nyc_model_output_path, nyc_fips):

    us_latest = combined_datasets.build_us_latest_with_all_fields()
    nyc_latest = us_latest.get_record_for_fips(nyc_fips)
    model_output = None
    expected_projections = None

    intervention = Intervention.OBSERVED_INTERVENTION
    if include_projections:
        model_output = CANPyseirLocationOutput.load_from_path(nyc_model_output_path)
        expected_projections = _Projections(
            totalHospitalBeds=_ResourceUsageProjection(
                peakShortfall=0, peakDate=datetime.date(2020, 4, 15), shortageStartDate=None
            ),
            ICUBeds=None,
            Rt=model_output.latest_rt,
            RtCI90=model_output.latest_rt_ci90,
        )
        intervention = Intervention.STRONG_INTERVENTION

    summary = generate_api.generate_area_summary(nyc_latest, model_output)

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
                "currentUsageCovid": 0,
                "currentUsageTotal": None,
                "typicalUsageRate": nyc_latest["all_beds_occupancy_rate"],
            },
            ICUBeds={
                "capacity": nyc_latest["icu_beds"],
                "totalCapacity": nyc_latest["icu_beds"],
                "currentUsageCovid": 0,
                "currentUsageTotal": 0,
                "typicalUsageRate": nyc_latest["icu_occupancy_rate"],
            },
            contactTracers=nyc_latest["contact_tracers_count"],
        ),
        lastUpdatedDate=datetime.datetime.utcnow(),
        projections=expected_projections,
    )
    import pprint

    pprint.pprint(expected.actuals.ICUBeds.dict())
    pprint.pprint(summary.actuals.ICUBeds.dict())
    assert expected.dict() == summary.dict()


@pytest.mark.parametrize("include_projections", [True])
def test_generate_timeseries_for_fips(include_projections, nyc_model_output_path, nyc_fips):

    us_latest = combined_datasets.build_us_latest_with_all_fields()
    us_timeseries = combined_datasets.build_us_timeseries_with_all_fields()

    nyc_latest = us_latest.get_record_for_fips(nyc_fips)
    nyc_timeseries = us_timeseries.get_subset(None, fips=nyc_fips)
    intervention = Intervention.OBSERVED_INTERVENTION
    model_output = CANPyseirLocationOutput.load_from_path(nyc_model_output_path)

    area_summary = generate_api.generate_area_summary(nyc_latest, model_output)
    area_timeseries = generate_api.generate_area_timeseries(
        area_summary, nyc_timeseries, model_output
    )

    summary = generate_api.generate_area_summary(nyc_latest, model_output)

    assert summary.dict() == area_timeseries.area_summary.dict()
    # Double checking that serialized json does not contain NaNs, all values should
    # be serialized using the simplejson wrapper.
    assert "NaN" not in area_timeseries.json()
