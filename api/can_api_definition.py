from typing import List, Optional
import pydantic
import datetime

"""
CovidActNow API

Documentation at https://github.com/covid-projections/covid-data-model/tree/master/api
"""


class _ResourceUsageProjection(pydantic.BaseModel):
    peakShortfall: int = pydantic.Field(
        ..., description="Shortfall of resource needed at the peek utilization"
    )
    peakDate: Optional[datetime.date] = pydantic.Field(
        ..., description="Date of peak resource utilization"
    )
    shortageStartDate: Optional[datetime.date] = pydantic.Field(
        ..., description="Date when resource shortage begins"
    )


class _Projections(pydantic.BaseModel):
    totalHospitalBeds: _ResourceUsageProjection = pydantic.Field(
        ..., description="Projection about total hospital bed utilization"
    )
    ICUBeds: Optional[_ResourceUsageProjection] = pydantic.Field(
        ..., description="Projection about ICU hospital bed utilization"
    )
    Rt: float = pydantic.Field(..., description="Historical or Inferred Rt")
    RtCI90: float = pydantic.Field(..., description="Rt standard deviation")


class _ResourceUtilization(pydantic.BaseModel):
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


class _Actuals(pydantic.BaseModel):
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
    hospitalBeds: Optional[_ResourceUtilization] = pydantic.Field(...)
    ICUBeds: Optional[_ResourceUtilization] = pydantic.Field(...)
    # contactTracers count is available for states, not counties.
    contactTracers: Optional[int] = pydantic.Field(default=None, description="# of Contact Tracers")


class CovidActNowAreaSummary(pydantic.BaseModel):
    countryName: str = "US"
    fips: str = pydantic.Field(..., description="Fips for State + County. Five character code")
    lat: float = pydantic.Field(..., description="Latitude of point within the state or county")
    long: float = pydantic.Field(..., description="Longitude of point within the state or county")
    lastUpdatedDate: datetime.date = pydantic.Field(..., description="Date of latest data")
    projections: Optional[_Projections] = pydantic.Field(...)
    actuals: Optional[_Actuals] = pydantic.Field(...)
    population: int = pydantic.Field(..., description="Total Population in geographic area.", gt=0)


# TODO(igor): countyName *must* be None
class CovidActNowStateSummary(CovidActNowAreaSummary):
    stateName: str = pydantic.Field(..., description="The state name")
    countyName: Optional[str] = pydantic.Field(default=None, description="The county name")


class CovidActNowCountySummary(CovidActNowAreaSummary):
    stateName: str = pydantic.Field(..., description="The state name")
    countyName: str = pydantic.Field(..., description="The county name")


class CANActualsTimeseriesRow(_Actuals):
    date: datetime.date = pydantic.Field(..., descrition="Date of timeseries data point")


class CANPredictionTimeseriesRow(pydantic.BaseModel):
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
    cumulativePositiveTests: Optional[int] = pydantic.Field(
        ..., description="Number of positive test results to date"
    )
    cumulativeNegativeTests: Optional[int] = pydantic.Field(
        ..., description="Number of negative test results to date"
    )


class PredictionTimeseriesRowWithHeader(CANPredictionTimeseriesRow):
    countryName: str = "US"
    stateName: str = pydantic.Field(..., description="The state name")
    countyName: Optional[str] = pydantic.Field(..., description="The county name")
    intervention: str = pydantic.Field(..., description="Name of high-level intervention in-place")
    fips: str = pydantic.Field(..., description="Fips for State + County. Five character code")
    lat: float = pydantic.Field(..., description="Latitude of point within the state or county")
    long: float = pydantic.Field(..., description="Longitude of point within the state or county")
    lastUpdatedDate: datetime.date = pydantic.Field(..., description="Date of latest data")


class CovidActNowStateTimeseries(CovidActNowStateSummary):
    timeseries: List[CANPredictionTimeseriesRow] = pydantic.Field(...)
    actualsTimeseries: List[CANActualsTimeseriesRow] = pydantic.Field(...)

    # pylint: disable=no-self-argument
    @pydantic.validator("timeseries")
    def check_timeseries_have_cumulative_test_data(cls, rows, values):
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


class CovidActNowCountyTimeseries(CovidActNowCountySummary):
    timeseries: List[CANPredictionTimeseriesRow] = pydantic.Field(...)
    actualsTimeseries: List[CANActualsTimeseriesRow] = pydantic.Field(...)


class CovidActNowCountiesAPI(pydantic.BaseModel):
    __root__: List[CovidActNowCountySummary] = pydantic.Field(...)


class CovidActNowStatesSummary(pydantic.BaseModel):
    __root__: List[CovidActNowStateSummary] = pydantic.Field(...)


class CovidActNowStatesTimeseries(pydantic.BaseModel):
    __root__: List[CovidActNowStateTimeseries] = pydantic.Field(...)


class CovidActNowCountiesSummary(pydantic.BaseModel):
    __root__: List[CovidActNowCountySummary] = pydantic.Field(...)


class CovidActNowCountiesTimeseries(pydantic.BaseModel):
    __root__: List[CovidActNowCountyTimeseries] = pydantic.Field(...)
