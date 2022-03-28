from typing import List, Optional, Dict
import enum
import textwrap

from libs.datasets.dataset_utils import AggregationLevel
from libs import base_model
from libs.datasets import timeseries
import pydantic
import datetime
from datapublic.common_fields import GetByValueMixin


CDC_TRANSMISSION_LEVEL_DESCRIPTION = textwrap.dedent(
    """
    Community transmission level for region, calculated using the CDC definition.

    Possible values:
        - 0: Low
        - 1: Moderate
        - 2: Substantial
        - 3: High
        - 4: Unknown

    See [definitions of CDC community transmission levels](
    https://covid.cdc.gov/covid-data-tracker/#cases_community) for more
    details.

    Note that the value may differ from what the CDC website reports
    given we have different data sources. We have also introduced an
    "Unknown" level for when both case data and test positivity data are
    missing for at least 15 days. The CDC does not have an "Unknown"
    level and instead will designate a location as "Low" when case and
    test positivity data are missing.
    """
)


class TestPositivityRatioMethod(GetByValueMixin, enum.Enum):
    """Method used to determine test positivity ratio."""

    CMSTesting = "CMSTesting"
    CDCTesting = "CDCTesting"
    HHSTesting = "HHSTesting"
    VALORUM = "Valorum"
    COVID_TRACKING = "covid_tracking"
    OTHER = "other"


class FieldSourceType(GetByValueMixin, enum.Enum):
    """The data source of a field (metric or actual). This enumeration lists the places from which
    CAN fetches data. The source is tracked on a per field and region timeseries basis."""

    NYTimes = "NYTimes"
    CMSTesting = "CMSTesting"
    CDCTesting = "CDCTesting"
    HHSTesting = "HHSTesting"
    HHSHospital = "HHSHospital"
    VALORUM = "Valorum"
    COVID_TRACKING = "covid_tracking"
    USA_FACTS = "USAFacts"
    TestAndTrace = "TestAndTrace"
    CANScrapersStateProviders = "CANScrapersStateProviders"
    OTHER = "other"


class TestPositivityRatioDetails(base_model.APIBaseModel):
    """Details about how the test positivity ratio was calculated."""

    source: TestPositivityRatioMethod = pydantic.Field(
        ..., description="Source data for test positivity ratio."
    )


class DemographicDistributions(base_model.APIBaseModel):
    """Distributions of demographic data.

    Note that different regions may have different demographic distributions for
    the same field.  For instance, health departments in different states may report
    different age ranges.

    The data provided matches the source distributions.
    """

    age: Optional[Dict[str, int]] = pydantic.Field(None)
    race: Optional[Dict[str, int]] = pydantic.Field(None)
    ethnicity: Optional[Dict[str, int]] = pydantic.Field(None)
    sex: Optional[Dict[str, int]] = pydantic.Field(None)


class HospitalResourceUtilization(base_model.APIBaseModel):
    capacity: Optional[int] = pydantic.Field(..., description="Total capacity for resource.")
    currentUsageTotal: Optional[int] = pydantic.Field(
        ..., description="Currently used capacity for resource by all patients (COVID + Non-COVID)"
    )
    currentUsageCovid: Optional[int] = pydantic.Field(
        ..., description="Currently used capacity for resource by COVID "
    )


class Actuals(base_model.APIBaseModel):
    """Known actuals data."""

    cases: Optional[int] = pydantic.Field(
        ..., description="Cumulative confirmed or suspected cases."
    )
    deaths: Optional[int] = pydantic.Field(
        ...,
        description=(
            "Cumulative deaths that are suspected or confirmed to have been caused by COVID-19."
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
        ...,
        description="""
Information about acute bed utilization details.

Fields:
 * capacity - Current staffed acute bed capacity.
 * currentUsageTotal - Total number of acute beds currently in use
 * currentUsageCovid - Number of acute beds currently in use by COVID patients.
""",
    )
    hsaHospitalBeds: Optional[HospitalResourceUtilization] = pydantic.Field(
        ...,
        description="""
Information about acute bed utilization details aggregated for the corresponding
Health Service Area.

Fields:
 * capacity - Current staffed acute bed capacity.
 * currentUsageTotal - Total number of acute beds currently in use
 * currentUsageCovid - Number of acute beds currently in use by COVID patients.
""",
    )
    icuBeds: Optional[HospitalResourceUtilization] = pydantic.Field(
        ...,
        description="""
Information about ICU bed utilization details.

Fields:
 * capacity - Current staffed ICU bed capacity.
 * currentUsageTotal - Total number of ICU beds currently in use
 * currentUsageCovid - Number of ICU beds currently in use by COVID patients.
""",
    )
    hsaIcuBeds: Optional[HospitalResourceUtilization] = pydantic.Field(
        ...,
        description="""
Information about ICU bed utilization details aggregated for the corresponding
Health Service Area.

Fields:
 * capacity - Current staffed ICU bed capacity.
 * currentUsageTotal - Total number of ICU beds currently in use
 * currentUsageCovid - Number of ICU beds currently in use by COVID patients.
""",
    )
    newCases: Optional[int] = pydantic.Field(
        ...,
        description="""
New confirmed or suspected cases.


New cases are a processed timeseries of cases - summing new cases may not equal
the cumulative case count.

Processing steps:
 1. If a region does not report cases for a period of time but then begins reporting again,
    we will exclude the first day that reporting recommences. This first day likely includes
    multiple days worth of cases and can be misleading to the overall series.
 2. We remove any days with negative new cases.
 3. We apply an outlier detection filter to the timeseries, which removes any data
    points that seem improbable given recent numbers. Many times this is due to
    backfill of previously unreported cases.
""",
    )
    newDeaths: Optional[int] = pydantic.Field(
        ...,
        description="""
New confirmed or suspected COVID-19 deaths.

New deaths is an estimate of deaths per day; summing new deaths may not equal the
cumulative death count.

Processing steps:
 1. If a region does not report deaths for a period of time but then begins reporting again,
    we will exclude the first day that reporting recommences. This first day likely includes
    multiple days worth of deaths and can be misleading to the overall series.
 2. We remove any days with negative new deaths.
 3. We apply an outlier detection filter to the timeseries, which removes any data
    points that seem improbable given recent numbers. Many times this is due to
    backfill of previously unreported deaths.
""",
    )
    vaccinesDistributed: Optional[int] = pydantic.Field(
        None, description="Number of vaccine doses distributed."
    )
    vaccinationsInitiated: Optional[int] = pydantic.Field(
        None,
        description="""
Number of vaccinations initiated.

This value may vary by type of vaccine, but for Moderna and Pfizer this indicates
number of people vaccinated with the first dose.
""",
    )
    vaccinationsCompleted: Optional[int] = pydantic.Field(
        None,
        description="""
Number of vaccinations completed.

This value may vary by type of vaccine, but for Moderna and Pfizer this indicates
number of people vaccinated with both the first and second dose.
""",
    )
    vaccinationsAdditionalDose: Optional[int] = pydantic.Field(
        None,
        description="Number of individuals who are fully vaccinated and have received a booster (or additional) dose.",
    )
    vaccinesAdministered: Optional[int] = pydantic.Field(
        None, description="Total number of vaccine doses administered."
    )
    vaccinesAdministeredDemographics: Optional[DemographicDistributions] = pydantic.Field(
        None, description="Demographic distributions for administered vaccines."
    )
    vaccinationsInitiatedDemographics: Optional[DemographicDistributions] = pydantic.Field(
        None, description="Demographic distributions for initiated vaccinations."
    )
    # When adding a new "actual" field here remember to add a `FieldAnnotations` in `Annotations`.


class ActualsTimeseriesRow(Actuals):
    """Actual data for a specific day."""

    date: datetime.date = pydantic.Field(..., description="Date of timeseries data point")


class AnomalyAnnotation(base_model.APIBaseModel):
    date: datetime.date = pydantic.Field(..., description="Date of anomaly")
    type: timeseries.TagType = pydantic.Field(..., description="Type of annotation")
    original_observation: float = pydantic.Field(
        ..., description="Original value on this date detected as anomalous."
    )


class FieldSource(base_model.APIBaseModel):
    type: Optional[FieldSourceType] = pydantic.Field(
        None, description="The type of data source from a CAN list of data source types"
    )
    url: Optional[str] = pydantic.Field(
        None, description="URL of a webpage containing the data at the source"
    )
    name: Optional[str] = pydantic.Field(None, description="A human readable name of the source")


class FieldAnnotations(base_model.APIBaseModel):
    """Annotations associated with one field."""

    sources: List[FieldSource]
    anomalies: List[AnomalyAnnotation]


class Annotations(base_model.APIBaseModel):
    """Annotations for each field."""

    # Keep this list of fields in sync with the fields in `Actuals`
    cases: Optional[FieldAnnotations] = pydantic.Field(None, description="Annotations for cases")
    deaths: Optional[FieldAnnotations] = pydantic.Field(None, description="Annotations for deaths")
    positiveTests: Optional[FieldAnnotations] = pydantic.Field(
        None, description="Annotations for positiveTests"
    )
    negativeTests: Optional[FieldAnnotations] = pydantic.Field(
        None, description="Annotations for negativeTests"
    )
    contactTracers: Optional[FieldAnnotations] = pydantic.Field(
        None, description="Annotations for contactTracers"
    )
    hospitalBeds: Optional[FieldAnnotations] = pydantic.Field(
        None, description="Annotations for hospitalBeds"
    )
    hsaHospitalBeds: Optional[FieldAnnotations] = pydantic.Field(
        None, description="Annotations for hsaHospitalBeds"
    )
    icuBeds: Optional[FieldAnnotations] = pydantic.Field(
        None, description="Annotations for icuBeds"
    )
    hsaIcuBeds: Optional[FieldAnnotations] = pydantic.Field(
        None, description="Annotations for hsaIcuBeds"
    )
    newCases: Optional[FieldAnnotations] = pydantic.Field(
        None, description="Annotations for newCases"
    )
    newDeaths: Optional[FieldAnnotations] = pydantic.Field(
        None, description="Annotations for newDeaths"
    )
    vaccinesDistributed: Optional[FieldAnnotations] = pydantic.Field(
        None, description="Annotations for vaccinesDistributed"
    )
    vaccinationsInitiated: Optional[FieldAnnotations] = pydantic.Field(
        None, description="Annotations for vaccinationsInitiated"
    )
    vaccinationsCompleted: Optional[FieldAnnotations] = pydantic.Field(
        None, description="Annotations for vaccinationsCompleted"
    )
    vaccinationsAdditionalDose: Optional[FieldAnnotations] = pydantic.Field(
        None, description="Annotations for vaccinationsAdditionalDose"
    )
    vaccinesAdministered: Optional[FieldAnnotations] = pydantic.Field(
        None, description="Annotations for vaccinesAdministered"
    )
    # Keep this list of fields in sync with the fields in `Metrics`
    testPositivityRatio: Optional[FieldAnnotations] = pydantic.Field(
        None, description="Annotations for testPositivityRatio"
    )
    caseDensity: Optional[FieldAnnotations] = pydantic.Field(
        None, description="Annotations for caseDensity"
    )
    weeklyNewCasesPer100k: Optional[FieldAnnotations] = pydantic.Field(
        None, description="Annotations for weeklyNewCasesPer100k"
    )
    contactTracerCapacityRatio: Optional[FieldAnnotations] = pydantic.Field(
        None, description="Annotations for contactTracerCapacityRatio"
    )
    infectionRate: Optional[FieldAnnotations] = pydantic.Field(
        None, description="Annotations for infectionRate"
    )
    infectionRateCI90: Optional[FieldAnnotations] = pydantic.Field(
        None, description="Annotations for infectionRateCI90"
    )
    icuCapacityRatio: Optional[FieldAnnotations] = pydantic.Field(
        None, description="Annotations for icuCapacityRatio"
    )
    vaccinationsInitiatedRatio: Optional[FieldAnnotations] = pydantic.Field(
        None, description="Annotations for vaccinationsInitiatedRatio"
    )
    vaccinationsCompletedRatio: Optional[FieldAnnotations] = pydantic.Field(
        None, description="Annotations for vaccinationsCompletedRatio"
    )
    vaccinationsAdditionalDoseRatio: Optional[FieldAnnotations] = pydantic.Field(
        None,
        description=(
            "Ratio of population that are fully vaccinated and have received a booster (or additional) dose."
        ),
    )


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

    weeklyNewCasesPer100k: Optional[float] = pydantic.Field(
        ..., description="The number of new cases per 100k population over the last week.",
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
    icuCapacityRatio: Optional[float] = pydantic.Field(
        ...,
        description="Ratio of staffed intensive care unit (ICU) beds that are currently in use.",
    )

    vaccinationsInitiatedRatio: Optional[float] = pydantic.Field(
        None, description=("Ratio of population that has initiated vaccination.")
    )
    vaccinationsCompletedRatio: Optional[float] = pydantic.Field(
        None, description=("Ratio of population that has completed vaccination.")
    )
    vaccinationsAdditionalDoseRatio: Optional[float] = pydantic.Field(
        None,
        description=(
            "Ratio of population that are fully vaccinated and have received a booster (or additional) dose."
        ),
    )

    @staticmethod
    def empty():
        """Returns an empty Metrics object."""
        return Metrics(
            testPositivityRatio=None,
            caseDensity=None,
            weeklyNewCasesPer100k=None,
            contactTracerCapacityRatio=None,
            infectionRate=None,
            infectionRateCI90=None,
            icuCapacityRatio=None,
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


@enum.unique
class CDCTransmissionLevel(enum.Enum):
    """CDC community transmission level."""

    LOW = 0

    MODERATE = 1

    SUBSTANTIAL = 2

    HIGH = 3

    UNKNOWN = 4


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
    icuCapacityRatio: RiskLevel = pydantic.Field(..., description="ICU capacity ratio risk level.")

    @classmethod
    def empty(cls) -> "RiskLevels":
        return RiskLevels(
            overall=RiskLevel.LOW,
            testPositivityRatio=RiskLevel.LOW,
            caseDensity=RiskLevel.LOW,
            contactTracerCapacityRatio=RiskLevel.LOW,
            infectionRate=RiskLevel.LOW,
            icuCapacityRatio=RiskLevel.LOW,
        )


# Additional class used for bulk timeseries where we are not including all risk levels
# right now, only the overall risk level.
class RiskLevelsRow(base_model.APIBaseModel):
    overall: RiskLevel = pydantic.Field(..., description="Overall risk level for region.")
    caseDensity: RiskLevel = pydantic.Field(..., description="Case density risk level for region.")


class RiskLevelTimeseriesRow(RiskLevelsRow):
    """Timeseries data for risk levels. Currently only surfacing overall risk level for region."""

    date: datetime.date = pydantic.Field(..., description="Date of timeseries data point")


class MetricsTimeseriesRow(Metrics):
    """Metrics data for a specific day."""

    date: datetime.date = pydantic.Field(..., description="Date of timeseries data point")


class CdcTransmissionLevelTimeseriesRow(base_model.APIBaseModel):
    date: datetime.date = pydantic.Field(..., description="Date of timeseries data point")
    cdcTransmissionLevel: CDCTransmissionLevel = pydantic.Field(
        ..., description=CDC_TRANSMISSION_LEVEL_DESCRIPTION
    )


class RegionSummary(base_model.APIBaseModel):
    """Summary of actual and prediction data for a single region."""

    fips: str = pydantic.Field(
        ...,
        description=(
            "FIPS Code. FIPS codes are either 2-digit state codes, "
            "5-digit county codes, 5-digit CBSA codes, or 1-digit '0' for the entire USA."
        ),
    )
    country: str = pydantic.Field(..., description="2-letter ISO-3166 Country code.")
    state: Optional[str] = pydantic.Field(
        ..., description="2-letter ANSI state code. For CBSA regions, state is omitted."
    )
    county: Optional[str] = pydantic.Field(..., description="County name")

    hsa: Optional[str] = pydantic.Field(
        ...,
        description="3 digit Health Service Area identification code. For CBSA, state, and country regions hsa is omitted. For more on HSAs see: https://github.com/covid-projections/covid-data-model/blob/main/data/misc/README.md",
    )
    level: AggregationLevel = pydantic.Field(..., description="Level of region.")
    lat: Optional[float] = pydantic.Field(
        ..., description="Latitude of point within the state or county. Currently a placeholder."
    )
    locationId: str = pydantic.Field(
        ...,
        description="Location ID as defined here: https://github.com/covidatlas/li/blob/master/docs/reports-v1.md#general-notes",
    )
    long: Optional[float] = pydantic.Field(
        ..., description="Longitude of point within the state or county. Currently a placeholder."
    )
    population: int = pydantic.Field(
        ..., description="Total Population in geographic region.", gt=0
    )

    hsaPopulation: Optional[int] = pydantic.Field(
        ...,
        description="Total Population of county's corresponding Health Service Area. For CBSA, state, and country regions hsaPopulation is omitted. For more on HSAs see: https://github.com/covid-projections/covid-data-model/blob/main/data/misc/README.md",
        gt=0,
    )

    metrics: Metrics = pydantic.Field(...)
    riskLevels: RiskLevels = pydantic.Field(..., description="Risk levels for region.")

    cdcTransmissionLevel: CDCTransmissionLevel = pydantic.Field(
        ..., description=CDC_TRANSMISSION_LEVEL_DESCRIPTION
    )

    actuals: Actuals = pydantic.Field(...)
    annotations: Annotations = pydantic.Field(...)

    lastUpdatedDate: datetime.date = pydantic.Field(..., description="Date of latest data")

    url: Optional[str] = pydantic.Field(
        ..., description="URL linking to Covid Act Now location page."
    )


class RegionSummaryWithTimeseries(RegionSummary):
    """Summary data for a region with prediction timeseries data and actual timeseries data."""

    metricsTimeseries: List[MetricsTimeseriesRow] = pydantic.Field(...)
    actualsTimeseries: List[ActualsTimeseriesRow] = pydantic.Field(...)
    riskLevelsTimeseries: List[RiskLevelTimeseriesRow] = pydantic.Field(...)
    cdcTransmissionLevelTimeseries: List[CdcTransmissionLevelTimeseriesRow] = pydantic.Field(...)

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
        description=(
            "FIPS Code. FIPS codes are either 2-digit state codes, "
            "5-digit county codes, 5-digit CBSA codes, or 1-digit '0' for the entire USA."
        ),
    )
    lat: Optional[float] = pydantic.Field(
        ..., description="Latitude of point within the state or county"
    )
    long: Optional[float] = pydantic.Field(
        ..., description="Longitude of point within the state or county"
    )
    locationId: str = pydantic.Field(
        ...,
        description="Location ID as defined here: https://github.com/covidatlas/li/blob/master/docs/reports-v1.md#general-notes",
    )
    actuals: Optional[Actuals] = pydantic.Field(..., description="Actuals for given day")
    metrics: Optional[Metrics] = pydantic.Field(..., description="Metrics for given day")
    riskLevels: Optional[RiskLevelsRow] = pydantic.Field(
        ..., description="Risk Levels for given day"
    )
    cdcTransmissionLevel: Optional[CDCTransmissionLevel] = pydantic.Field(
        ..., description=CDC_TRANSMISSION_LEVEL_DESCRIPTION
    )

    hsa: Optional[str] = pydantic.Field(
        ...,
        description="3 digit Health Service Area identification code. For CBSA, state, and country regions hsa is omitted. For more on HSAs see: https://github.com/covid-projections/covid-data-model/blob/main/data/misc/README.md",
    )

    hsaPopulation: Optional[int] = pydantic.Field(
        ...,
        description="Total Population of county's corresponding Health Service Area. For CBSA, state, and country regions hsaPopulation is omitted. For more on HSAs see: https://github.com/covid-projections/covid-data-model/blob/main/data/misc/README.md",
        gt=0,
    )


class AggregateFlattenedTimeseries(base_model.APIBaseModel):
    """Flattened timeseries data for multiple regions."""

    __root__: List[RegionTimeseriesRowWithHeader] = pydantic.Field(...)
