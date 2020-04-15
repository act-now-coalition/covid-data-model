import unittest
from datetime import date
from api.can_api_definition import (
    CovidActNowCountiesAPI,
    CovidActNowCountySummary,
)


class CovidTimeseriesModelTest(unittest.TestCase):
    def test_counties_api(self):
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
            },
            actuals={
                "population": 883305,
                "intervention": "stay_at_home",
                "cumulativeConfirmedCases": 10,
                "cumulativeDeaths": 5,
                "hospitalBeds": {"capacity": 100, "currentUsage": 3},
                "ICUBeds": {"capacity": 10, "currentUsage": 2},
            },
        )
        counties = CovidActNowCountiesAPI(data=[county_summary])


if __name__ == "__main__":
    unittest.main()
