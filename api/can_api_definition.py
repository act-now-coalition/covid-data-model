from typing import List, Optional
from libs.enums import Intervention
from libs.datasets.dataset_utils import AggregationLevel
from libs import us_state_abbrev
from libs import base_model
import pydantic
import datetime

"""
CovidActNow API

Documentation at https://github.com/covid-projections/covid-data-model/tree/master/api
"""


class ResourceUsageProjection(base_model.APIBaseModel):
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
    totalHospitalBeds: ResourceUsageProjection = pydantic.Field(
        ..., description="Projection about total hospital bed utilization"
    )
    ICUBeds: Optional[ResourceUsageProjection] = pydantic.Field(
        ..., description="Projection about ICU hospital bed utilization"
    )
    Rt: float = pydantic.Field(..., description="Historical or Inferred Rt")
    RtCI90: float = pydantic.Field(..., description="Rt standard deviation")


class ResourceUtilization(base_model.APIBaseModel):
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
        ..., description="Currently used capacity for resource by all patients (COVID + Non-COVID)",
    )
    typicalUsageRate: Optional[float] = pydantic.Field(
        ..., description="Typical used capacity rate for resource. This excludes any COVID usage.",
    )


class Actuals(base_model.APIBaseModel):
    population: Optional[int] = pydantic.Field(
        ...,
        description="Total population in geographic area [*deprecated*: refer to summary for this]",
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


class CovidActNowAreaSummary(base_model.APIBaseModel):
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
    population: int = pydantic.Field(..., description="Total Population in geographic area.", gt=0)

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


class CANActualsTimeseriesRow(Actuals):
    date: datetime.date = pydantic.Field(..., descrition="Date of timeseries data point")


class CANPredictionTimeseriesRow(base_model.APIBaseModel):
    date: datetime.date = pydantic.Field(..., descrition="Date of timeseries data point")
    hospitalBedsRequired: int = pydantic.Field(
        ...,
        description="Number of hospital beds projected to be in-use or that were actually in use (if in the past)",
    )
    hospitalBedCapacity: int = pydantic.Field(
        ...,
        description="Number of hospital beds projected to be in-use or actually in use (if in the past)",
    )
    ICUBedsInUse: int = pydantic.Field(
        ...,
        description="Number of ICU beds projected to be in-use or that were actually in use (if in the past)",
    )
    ICUBedCapacity: int = pydantic.Field(
        ...,
        description="Number of ICU beds projected to be in-use or actually in use (if in the past)",
    )
    ventilatorsInUse: int = pydantic.Field(
        ..., description="Number of ventilators projected to be in-use.",
    )
    ventilatorCapacity: int = pydantic.Field(..., description="Total ventilator capacity.")
    RtIndicator: float = pydantic.Field(..., description="Historical or Inferred Rt")
    RtIndicatorCI90: float = pydantic.Field(..., description="Rt standard deviation")
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


class PredictionTimeseriesRowWithHeader(CANPredictionTimeseriesRow):
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


class CovidActNowAreaTimeseries(CovidActNowAreaSummary):
    timeseries: Optional[List[CANPredictionTimeseriesRow]] = pydantic.Field(...)
    actualsTimeseries: List[CANActualsTimeseriesRow] = pydantic.Field(...)

    @property
    def area_summary(self) -> CovidActNowAreaSummary:

        data = {}
        # Iterating through self does not force any conversion
        # https://pydantic-docs.helpmanual.io/usage/exporting_models/#dictmodel-and-iteration
        for field, value in self:
            if field not in CovidActNowAreaSummary.__fields__:
                continue
            data[field] = value

        return CovidActNowAreaSummary(**data)

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


class CovidActNowBulkSummary(base_model.APIBaseModel):
    __root__: List[CovidActNowAreaSummary] = pydantic.Field(...)

    def output_key(self, intervention):
        aggregate_level = self.__root__[0].aggregate_level
        if aggregate_level is AggregationLevel.COUNTY:
            return f"counties.{intervention.name}"
        if aggregate_level is AggregationLevel.STATE:
            return f"states.{intervention.name}"


class CovidActNowBulkTimeseries(base_model.APIBaseModel):
    __root__: List[CovidActNowAreaTimeseries] = pydantic.Field(...)

    def output_key(self, intervention):
        aggregate_level = self.__root__[0].aggregate_level
        if aggregate_level is AggregationLevel.COUNTY:
            return f"counties.{intervention.name}.timeseries"
        if aggregate_level is AggregationLevel.STATE:
            return f"states.{intervention.name}.timeseries"


class CovidActNowBulkFlattenedTimeseries(base_model.APIBaseModel):
    __root__: List[PredictionTimeseriesRowWithHeader] = pydantic.Field(...)

    def output_key(self, intervention):
        aggregate_level = self.__root__[0].aggregate_level
        if aggregate_level is AggregationLevel.COUNTY:
            return f"counties.{intervention.name}.timeseries"
        if aggregate_level is AggregationLevel.STATE:
            return f"states.{intervention.name}.timeseries"
