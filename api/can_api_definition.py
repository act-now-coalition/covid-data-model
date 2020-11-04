"""
CovidActNow API

Documentation at https://github.com/covid-projections/covid-data-model/tree/master/api
"""


from typing import List, Optional
import enum
import datetime
import pydantic
from covidactnow.datapublic.common_fields import GetByValueMixin

from libs import base_model
from libs import us_state_abbrev
from libs.datasets.dataset_utils import AggregationLevel
from libs.enums import Intervention


class ResourceUsageProjection(base_model.APIBaseModel):
    """Resource usage projection data."""

    peakShortfall: int = pydantic.Field(
        ..., description="Shortfall of resource needed at the peak utilization"
    )
    peakDate: Optional[datetime.date] = pydantic.Field(
        ..., description="Date of peak resource utilization"
    )
    shortageStartDate: Optional[datetime.date] = pydantic.Field(
        ..., description="Date when resource shortage begins"
    )


class Projections(base_model.APIBaseModel):
    """Summary of projection data."""

    totalHospitalBeds: ResourceUsageProjection = pydantic.Field(
        ..., description="Projection about total hospital bed utilization"
    )
    ICUBeds: Optional[ResourceUsageProjection] = pydantic.Field(
        ..., description="Projection about ICU hospital bed utilization"
    )
    Rt: Optional[float] = pydantic.Field(..., description="Inferred Rt")
    RtCI90: Optional[float] = pydantic.Field(
        ..., description="Rt 90th percentile confidence interval upper endpoint."
    )


class ResourceUtilization(base_model.APIBaseModel):
    """Utilization of hospital resources."""

    capacity: Optional[int] = pydantic.Field(
        ...,
        description=(
            "*deprecated*: Capacity for resource. In the case of ICUs, "
            "this refers to total capacity. For hospitalization this refers to free capacity for "
            "COVID patients. This value is calculated by (1 - typicalUsageRate) * totalCapacity * 2.07"
        ),
    )
    totalCapacity: Optional[int] = pydantic.Field(..., description="Total capacity for resource.")
    currentUsageCovid: Optional[int] = pydantic.Field(
        ..., description="Currently used capacity for resource by COVID "
    )
    currentUsageTotal: Optional[int] = pydantic.Field(
        ..., description="Currently used capacity for resource by all patients (COVID + Non-COVID)"
    )
    typicalUsageRate: Optional[float] = pydantic.Field(
        ..., description="Typical used capacity rate for resource. This excludes any COVID usage."
    )


class CovidPatientsMethod(enum.Enum):
    """Method used to determine number of current ICU patients with covid."""

    ACTUAL = "actual"
    ESTIMATED = "estimated"


class TestPositivityRatioMethod(GetByValueMixin, enum.Enum):
    """Method used to determine test positivity ratio."""

    CMSTesting = "CMSTesting"
    HHSTesting = "HHSTesting"
    VALORUM = "Valorum"
    OTHER = "other"


class NonCovidPatientsMethod(enum.Enum):
    """Method used to determine number of current ICU patients without covid."""

    ACTUAL = "actual"
    ESTIMATED_FROM_TYPICAL_UTILIZATION = "estimated_from_typical_utilization"
    ESTIMATED_FROM_TOTAL_ICU_ACTUAL = "estimated_from_total_icu_actual"


class TestPositivityRatioDetails(base_model.APIBaseModel):
    """Details about how the test positivity ratio was calculated."""

    source: TestPositivityRatioMethod = pydantic.Field(
        ..., description="Source data for test positivity ratio."
    )


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
        ..., description="90th percentile confidence interval upper endpoint of the infection rate."
    )
    icuHeadroomRatio: Optional[float] = pydantic.Field(...)
    icuHeadroomDetails: Optional[ICUHeadroomMetricDetails] = pydantic.Field(None)


class MetricsTimeseriesRow(Metrics):
    """Metrics data for a single day."""

    date: datetime.date = pydantic.Field(..., descrition="Date of timeseries data point")


class Actuals(base_model.APIBaseModel):
    """Known actuals data."""

    population: Optional[int] = pydantic.Field(
        ...,
        description="Total population in geographic region [*deprecated*: refer to summary for this]",
        gt=0,
    )
    intervention: str = pydantic.Field(..., description="Name of high-level intervention in-place")
    cumulativeConfirmedCases: Optional[int] = pydantic.Field(
        ..., description="Number of confirmed cases so far"
    )
    cumulativePositiveTests: Optional[int] = pydantic.Field(
        ..., description="Number of positive test results to date"
    )
    cumulativeNegativeTests: Optional[int] = pydantic.Field(
        ..., description="Number of negative test results to date"
    )
    cumulativeDeaths: Optional[int] = pydantic.Field(..., description="Number of deaths so far")
    hospitalBeds: Optional[ResourceUtilization] = pydantic.Field(...)
    ICUBeds: Optional[ResourceUtilization] = pydantic.Field(...)
    # contactTracers count is available for states, not counties.
    contactTracers: Optional[int] = pydantic.Field(default=None, description="# of Contact Tracers")


class RegionSummary(base_model.APIBaseModel):
    """Summary of actual and prediction data for a single region."""

    countryName: str = "US"
    fips: str = pydantic.Field(
        ...,
        description="Fips Code.  For state level data, 2 characters, for county level data, 5 characters.",
    )
    lat: Optional[float] = pydantic.Field(
        ..., description="Latitude of point within the state or county"
    )
    long: Optional[float] = pydantic.Field(
        ..., description="Longitude of point within the state or county"
    )
    stateName: str = pydantic.Field(..., description="The state name")
    countyName: Optional[str] = pydantic.Field(default=None, description="The county name")
    lastUpdatedDate: datetime.date = pydantic.Field(..., description="Date of latest data")
    projections: Optional[Projections] = pydantic.Field(...)
    actuals: Optional[Actuals] = pydantic.Field(...)
    metrics: Optional[Metrics] = pydantic.Field(default=None, description="Region level metrics")
    population: int = pydantic.Field(
        ..., description="Total Population in geographic region.", gt=0
    )

    @property
    def intervention(self) -> Optional[Intervention]:
        if not self.actuals:
            return None

        return Intervention[self.actuals.intervention]

    @property
    def aggregate_level(self) -> AggregationLevel:
        if len(self.fips) == 2:
            return AggregationLevel.STATE

        if len(self.fips) == 5:
            return AggregationLevel.COUNTY

    @property
    def state(self) -> str:
        """State abbreviation."""
        return us_state_abbrev.US_STATE_ABBREV[self.stateName]

    def output_key(self, intervention: Intervention):
        if self.aggregate_level is AggregationLevel.STATE:
            return f"{self.state}.{intervention.name}"

        if self.aggregate_level is AggregationLevel.COUNTY:
            return f"{self.fips}.{intervention.name}"


class ActualsTimeseriesRow(Actuals):
    """Actual data for a specific day."""

    date: datetime.date = pydantic.Field(..., descrition="Date of timeseries data point")


class PredictionTimeseriesRow(base_model.APIBaseModel):
    """Prediction data for a single day."""

    date: datetime.date = pydantic.Field(..., descrition="Date of timeseries data point")
    hospitalBedsRequired: int = pydantic.Field(
        ...,
        description="Number of hospital beds projected to be in-use or that were actually in use (if in the past)",
    )
    hospitalBedCapacity: int = pydantic.Field(
        ...,
        description="Number of hospital beds projected to be in-use or actually in use (if in the past)",
    )
    ICUBedsInUse: Optional[int] = pydantic.Field(
        ...,
        description="Number of ICU beds projected to be in-use or that were actually in use (if in the past)",
    )
    ICUBedCapacity: int = pydantic.Field(
        ...,
        description="Number of ICU beds projected to be in-use or actually in use (if in the past)",
    )
    ventilatorsInUse: int = pydantic.Field(
        ..., description="Number of ventilators projected to be in-use."
    )
    ventilatorCapacity: int = pydantic.Field(..., description="Total ventilator capacity.")
    RtIndicator: Optional[float] = pydantic.Field(..., description="Historical or Inferred Rt")
    RtIndicatorCI90: Optional[float] = pydantic.Field(..., description="Rt standard deviation")
    cumulativeDeaths: int = pydantic.Field(..., description="Number of cumulative deaths")
    cumulativeInfected: Optional[int] = pydantic.Field(
        ..., description="Number of cumulative infections"
    )
    currentInfected: Optional[int] = pydantic.Field(..., description="Number of current infections")
    currentSusceptible: Optional[int] = pydantic.Field(
        ..., description="Number of people currently susceptible "
    )
    currentExposed: Optional[int] = pydantic.Field(
        ..., description="Number of people currently exposed"
    )


class PredictionTimeseriesRowWithHeader(PredictionTimeseriesRow):
    """Prediction timeseries row with location information."""

    countryName: str = "US"
    stateName: str = pydantic.Field(..., description="The state name")
    countyName: Optional[str] = pydantic.Field(..., description="The county name")
    intervention: str = pydantic.Field(..., description="Name of high-level intervention in-place")
    fips: str = pydantic.Field(..., description="Fips for State + County. Five character code")
    lat: Optional[float] = pydantic.Field(
        ..., description="Latitude of point within the state or county"
    )
    long: Optional[float] = pydantic.Field(
        ..., description="Longitude of point within the state or county"
    )
    lastUpdatedDate: datetime.date = pydantic.Field(..., description="Date of latest data")

    @property
    def aggregate_level(self) -> AggregationLevel:
        if len(self.fips) == 2:
            return AggregationLevel.STATE

        if len(self.fips) == 5:
            return AggregationLevel.COUNTY


class RegionSummaryWithTimeseries(RegionSummary):
    """Summary data for a region with prediction timeseries data and actual timeseries data."""

    timeseries: Optional[List[PredictionTimeseriesRow]] = pydantic.Field(...)
    actualsTimeseries: List[ActualsTimeseriesRow] = pydantic.Field(...)
    metricsTimeseries: Optional[List[MetricsTimeseriesRow]] = pydantic.Field(...)

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

    # pylint: disable=no-self-argument
    @pydantic.validator("timeseries")
    def check_timeseries_have_cumulative_test_data(cls, rows, values):
        # TODO: Fix validation
        return rows
        # Nebraska is missing testing data.
        state_full_name = values["stateName"]
        if state_full_name == "Nebraska":
            return rows
        total_negative_tests = sum(row.cumulativeNegativeTests or 0 for row in rows)
        total_positive_tests = sum(row.cumulativePositiveTests or 0 for row in rows)

        if not total_positive_tests or not total_negative_tests:
            raise ValueError(f"Missing cumulative test data for {state_full_name}.")

        return rows

    @pydantic.validator("timeseries")
    def check_timeseries_one_row_per_date(cls, rows, values):
        dates_in_row = len(set(row.date for row in rows))
        if len(rows) != dates_in_row:

            raise ValueError(
                "Number of rows does not match number of dates: " f"{len(rows)} vs. {dates_in_row}"
            )

        return rows

    def output_key(self, intervention: Intervention) -> str:
        return super().output_key(intervention) + ".timeseries"


class AggregateRegionSummary(base_model.APIBaseModel):
    """Summary data for multiple regions."""

    __root__: List[RegionSummary] = pydantic.Field(...)

    def output_key(self, intervention):
        aggregate_level = self.__root__[0].aggregate_level
        if aggregate_level is AggregationLevel.COUNTY:
            return f"counties.{intervention.name}"
        if aggregate_level is AggregationLevel.STATE:
            return f"states.{intervention.name}"


class AggregateRegionSummaryWithTimeseries(base_model.APIBaseModel):
    """Timeseries and summary data for multiple regions."""

    __root__: List[RegionSummaryWithTimeseries] = pydantic.Field(...)

    def output_key(self, intervention):
        aggregate_level = self.__root__[0].aggregate_level
        if aggregate_level is AggregationLevel.COUNTY:
            return f"counties.{intervention.name}.timeseries"
        if aggregate_level is AggregationLevel.STATE:
            return f"states.{intervention.name}.timeseries"


class AggregateFlattenedTimeseries(base_model.APIBaseModel):
    """Flattened prediction timeseries data for multiple regions."""

    __root__: List[PredictionTimeseriesRowWithHeader] = pydantic.Field(...)

    def output_key(self, intervention):
        aggregate_level = self.__root__[0].aggregate_level
        if aggregate_level is AggregationLevel.COUNTY:
            return f"counties.{intervention.name}.timeseries"
        if aggregate_level is AggregationLevel.STATE:
            return f"states.{intervention.name}.timeseries"
