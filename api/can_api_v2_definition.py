from typing import List, Optional
import enum

from libs.datasets.dataset_utils import AggregationLevel
from libs import base_model
import pydantic
import datetime
from covidactnow.datapublic.common_fields import GetByValueMixin


class TestPositivityRatioMethod(GetByValueMixin, enum.Enum):
    """Method used to determine test positivity ratio."""

    CMSTesting = "CMSTesting"
    CDCTesting = "CDCTesting"
    HHSTesting = "HHSTesting"
    VALORUM = "Valorum"
    COVID_TRACKING = "covid_tracking"
    OTHER = "other"


class TestPositivityRatioDetails(base_model.APIBaseModel):
    """Details about how the test positivity ratio was calculated."""

    source: TestPositivityRatioMethod = pydantic.Field(
        ..., description="Source data for test positivity ratio."
    )


class CovidPatientsMethod(enum.Enum):
    """Method used to determine number of current ICU patients with covid."""

    ACTUAL = "actual"
    ESTIMATED = "estimated"


class NonCovidPatientsMethod(enum.Enum):
    """Method used to determine number of current ICU patients without covid."""

    ACTUAL = "actual"
    ESTIMATED_FROM_TYPICAL_UTILIZATION = "estimated_from_typical_utilization"
    ESTIMATED_FROM_TOTAL_ICU_ACTUAL = "estimated_from_total_icu_actual"


class ICUHeadroomMetricDetails(base_model.APIBaseModel):
    """Details about how the ICU Headroom Metric was calculated."""

    currentIcuCovid: int = pydantic.Field(
        ..., description="Current number of covid patients in icu."
    )
    currentIcuCovidMethod: CovidPatientsMethod = pydantic.Field(
        ..., description="Method used to determine number of current ICU patients with covid."
    )
    currentIcuNonCovid: int = pydantic.Field(
        ..., description="Current number of covid patients in icu."
    )
    currentIcuNonCovidMethod: NonCovidPatientsMethod = pydantic.Field(
        ..., description="Method used to determine number of current ICU patients without covid."
    )


class HospitalResourceUtilization(base_model.APIBaseModel):
    capacity: Optional[int] = pydantic.Field(..., description="Total capacity for resource.")
    currentUsageTotal: Optional[int] = pydantic.Field(
        ..., description="Currently used capacity for resource by all patients (COVID + Non-COVID)"
    )
    currentUsageCovid: Optional[int] = pydantic.Field(
        ..., description="Currently used capacity for resource by COVID "
    )
    typicalUsageRate: Optional[float] = pydantic.Field(
        ..., description="Typical used capacity rate for resource. This excludes any COVID usage."
    )


class Actuals(base_model.APIBaseModel):
    """Known actuals data."""

    cases: Optional[int] = pydantic.Field(
        ..., description="Cumulative number of confirmed or suspected cases"
    )
    deaths: Optional[int] = pydantic.Field(
        ...,
        description=(
            "Cumulative number of deaths that are suspected or "
            "confirmed to have been caused by COVID-19"
        ),
    )
    positiveTests: Optional[int] = pydantic.Field(
        ..., description="Cumulative positive test results to date"
    )
    negativeTests: Optional[int] = pydantic.Field(
        ..., description="Cumulative negative test results to date"
    )
    contactTracers: Optional[int] = pydantic.Field(..., description="Number of Contact Tracers")
    hospitalBeds: Optional[HospitalResourceUtilization] = pydantic.Field(
        ..., description="Information about hospital bed utilization"
    )
    icuBeds: Optional[HospitalResourceUtilization] = pydantic.Field(
        ..., description="Information about ICU bed utilization"
    )
    newCases: Optional[int] = pydantic.Field(
        ...,
        description="""
New confirmed or suspected cases.

New cases are a processed timeseries of cases - summing new cases may not equal
the cumulative case count.

Notable exceptions:
 1. If a region does not report cases for a period of time, the first day
    cases start reporting again will not be included. This day likely includes
    multiple days worth of cases and can be misleading to the overall series.
 2. Any days with negative new cases are removed.
""",
    )


class ActualsTimeseriesRow(Actuals):
    """Actual data for a specific day."""

    date: datetime.date = pydantic.Field(..., description="Date of timeseries data point")


class Metrics(base_model.APIBaseModel):
    """Calculated metrics data based on known actuals."""

    testPositivityRatio: Optional[float] = pydantic.Field(
        ...,
        description="Ratio of people who test positive calculated using a 7-day rolling average.",
    )
    testPositivityRatioDetails: Optional[TestPositivityRatioDetails] = pydantic.Field(None)

    caseDensity: Optional[float] = pydantic.Field(
        ...,
        description="The number of cases per 100k population calculated using a 7-day rolling average.",
    )

    contactTracerCapacityRatio: Optional[float] = pydantic.Field(
        ...,
        description=(
            "Ratio of currently hired tracers to estimated "
            "tracers needed based on 7-day daily case average."
        ),
    )

    infectionRate: Optional[float] = pydantic.Field(
        ..., description="R_t, or the estimated number of infections arising from a typical case."
    )

    infectionRateCI90: Optional[float] = pydantic.Field(
        ...,
        description="90th percentile confidence interval upper endpoint of the infection rate.",
    )
    icuHeadroomRatio: Optional[float] = pydantic.Field(...)
    icuHeadroomDetails: ICUHeadroomMetricDetails = pydantic.Field(None)

    @staticmethod
    def empty():
        """Returns an empty Metrics object."""
        return Metrics(
            testPositivityRatio=None,
            caseDensity=None,
            contactTracerCapacityRatio=None,
            infectionRate=None,
            infectionRateCI90=None,
            icuHeadroomRatio=None,
        )


@enum.unique
class RiskLevel(enum.Enum):
    """COVID Risk Level.

## Risk Level Definitions
 *Low* - On track to contain COVID
 *Medium* - Slow disease growth
 *High* - At risk of outbreak
 *Critical* - Active or imminent outbreak
 *Unknown* - Risk unknown
 *Extreme* - Severe outbreak
"""

    LOW = 0

    MEDIUM = 1

    HIGH = 2

    CRITICAL = 3

    UNKNOWN = 4

    EXTREME = 5


class RiskLevels(base_model.APIBaseModel):
    """COVID risk levels for a region."""

    overall: RiskLevel = pydantic.Field(..., description="Overall risk level for region.")
    testPositivityRatio: RiskLevel = pydantic.Field(
        ..., description="Test positivity ratio risk level."
    )
    caseDensity: RiskLevel = pydantic.Field(..., description="Case density risk level.")
    contactTracerCapacityRatio: RiskLevel = pydantic.Field(
        ..., description="Contact tracer capacity ratio risk level."
    )
    infectionRate: RiskLevel = pydantic.Field(..., description="Infection rate risk level.")
    icuHeadroomRatio: RiskLevel = pydantic.Field(..., description="ICU headroom ratio risk level.")


class MetricsTimeseriesRow(Metrics):
    """Metrics data for a specific day."""

    date: datetime.date = pydantic.Field(..., description="Date of timeseries data point")


class RegionSummary(base_model.APIBaseModel):
    """Summary of actual and prediction data for a single region."""

    fips: str = pydantic.Field(
        ...,
        description=(
            "FIPS Code. FIPS codes are either 2-digit state codes, "
            "5-digit county codes, or 5-digit CBSA codes."
        ),
    )
    country: str = pydantic.Field(..., description="2-letter ISO-3166 Country code.")
    state: Optional[str] = pydantic.Field(
        ..., description="2-letter ANSI state code. For CBSA regions, state is omitted."
    )
    county: Optional[str] = pydantic.Field(..., description="County name")

    level: AggregationLevel = pydantic.Field(..., description="Level of region.")
    lat: Optional[float] = pydantic.Field(
        ..., description="Latitude of point within the state or county"
    )
    locationId: str = pydantic.Field(
        ...,
        description="Location ID as defined here: https://github.com/covidatlas/li/blob/master/docs/reports-v1.md#general-notes",
    )
    long: Optional[float] = pydantic.Field(
        ..., description="Longitude of point within the state or county"
    )
    population: int = pydantic.Field(
        ..., description="Total Population in geographic region.", gt=0
    )
    metrics: Metrics = pydantic.Field(...)
    riskLevels: RiskLevels = pydantic.Field(..., description="Risk levels for region.")
    actuals: Actuals = pydantic.Field(...)

    lastUpdatedDate: datetime.date = pydantic.Field(..., description="Date of latest data")

    url: Optional[str] = pydantic.Field(
        ..., description="URL linking to Covid Act Now location page."
    )


class RegionSummaryWithTimeseries(RegionSummary):
    """Summary data for a region with prediction timeseries data and actual timeseries data."""

    metricsTimeseries: List[MetricsTimeseriesRow] = pydantic.Field(None)
    actualsTimeseries: List[ActualsTimeseriesRow] = pydantic.Field(...)

    @property
    def region_summary(self) -> RegionSummary:

        data = {}
        # Iterating through self does not force any conversion
        # https://pydantic-docs.helpmanual.io/usage/exporting_models/#dictmodel-and-iteration
        for field, value in self:
            if field not in RegionSummary.__fields__:
                continue
            data[field] = value

        return RegionSummary(**data)


class AggregateRegionSummary(base_model.APIBaseModel):
    """Summary data for multiple regions."""

    __root__: List[RegionSummary] = pydantic.Field(...)


class AggregateRegionSummaryWithTimeseries(base_model.APIBaseModel):
    """Timeseries and summary data for multiple regions."""

    __root__: List[RegionSummaryWithTimeseries] = pydantic.Field(...)


class RegionTimeseriesRowWithHeader(base_model.APIBaseModel):
    """Prediction timeseries row with location information."""

    date: datetime.date = pydantic.Field(..., description="Date of timeseries data point")
    country: str = pydantic.Field(..., description="2-letter ISO-3166 Country code.")
    state: Optional[str] = pydantic.Field(..., description="2-letter ANSI state code.")
    county: Optional[str] = pydantic.Field(..., description="County name")
    fips: str = pydantic.Field(
        ...,
        description="Fips Code.  For state level data, 2 characters, for county level data, 5 characters.",
    )
    lat: float = pydantic.Field(None, description="Latitude of point within the state or county")
    long: float = pydantic.Field(None, description="Longitude of point within the state or county")
    locationId: str = pydantic.Field(
        ...,
        description="Location ID as defined here: https://github.com/covidatlas/li/blob/master/docs/reports-v1.md#general-notes",
    )
    actuals: Optional[Actuals] = pydantic.Field(..., description="Actuals for given day")
    metrics: Optional[Metrics] = pydantic.Field(..., description="Metrics for given day")


class AggregateFlattenedTimeseries(base_model.APIBaseModel):
    """Flattened timeseries data for multiple regions."""

    __root__: List[RegionTimeseriesRowWithHeader] = pydantic.Field(...)
