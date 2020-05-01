from datetime import date
from api.can_api_definition import CovidActNowCountiesAPI
from api.can_api_definition import CovidActNowCountiesTimeseries
from api.can_api_definition import CovidActNowCountySummary

from libs.pipelines import api_pipeline


def test_counties_api_output():
    county_summary = CovidActNowCountySummary(
        countryName="US",
        stateName="California",
        countyName="San Francisco",
        fips="06075",
        lat=37.7749,
        long=122.4194,
        lastUpdatedDate=date.today(),
        projections={
            "totalHospitalBeds": {
                "peakShortfall": 10,
                "peakDate": date(2020, 5, 13),
                "shortageStartDate": None,
            },
            "ICUBeds": {
                "peakShortfall": 5,
                "peakDate": date(2020, 5, 11),
                "shortageStartDate": None,
            },
            "cumulativeDeaths": 5,
            "peakDeaths": 10,
            "peakDeathsDate": date(2020, 5, 15),
            "endDate": date(2020, 10, 1),
            "Rt": 1.1,
            "RtCI90": 0.1
        },
        actuals={
            "population": 883305,
            "intervention": "stay_at_home",
            "cumulativeConfirmedCases": 10,
            "cumulativeDeaths": 5,
            "cumulativePositiveTests": None,
            "cumulativeNegativeTests": 20,
            "hospitalBeds": {"capacity": 100, "currentUsage": 3, "typicalUsageRate": 0.4},
            "ICUBeds": {"capacity": 10, "currentUsage": 2, "typicalUsageRate": 0.6},
        },
    )
    counties = CovidActNowCountiesAPI(__root__=[county_summary])
    assert counties


def test_remove_root():
    value = CovidActNowCountiesTimeseries(__root__=[]).dict()
    assert value == {"__root__": []}
    result = api_pipeline.remove_root_wrapper(value)
    assert result == []
