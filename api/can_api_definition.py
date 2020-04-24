from typing import List, Optional
import pydantic
import datetime

"""
Documented https://www.dropbox.com/scl/fi/o4bec2kz8dkcxdtabtqda/CAN-V1-API-Draft.paper?dl=0&rlkey=f7elx3zmo6tt5s7mj5nvap9a0

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
    Rt: float = pydantic.Field(
        ..., description="Historical or Inferred Rt"
    )
    RtCI90: float = pydantic.Field(
        ..., description="Rt standard deviation"
    )


class _ResourceUtilization(pydantic.BaseModel):
    capacity: int = pydantic.Field(..., description="Total capacity for resource")
    currentUsage: Optional[int] = pydantic.Field(
        ..., description="Currently used capacity for resource"
    )


class _Actuals(pydantic.BaseModel):
    population: int = pydantic.Field(
        ..., description="Total population in geographic area", gt=0
    )
    intervention: str = pydantic.Field(
        ..., description="Name of high-level intervention in-place"
    )
    cumulativeConfirmedCases: int = pydantic.Field(
        ..., description="Number of confirmed cases so far"
    )
    cumulativePositiveTests: Optional[int] = pydantic.Field(
        ..., description="Number of positive test results to date"
    )
    cumulativeNegativeTests: Optional[int] = pydantic.Field(
        ..., description="Number of negative test results to date"
    )
    cumulativeDeaths: int = pydantic.Field(..., description="Number of deaths so far")
    hospitalBeds: _ResourceUtilization = pydantic.Field(...)
    ICUBeds: Optional[_ResourceUtilization] = pydantic.Field(...)


class CovidActNowAreaSummary(pydantic.BaseModel):
    countryName: str = "US"
    fips: str = pydantic.Field(
        ..., description="Fips for State + County. Five character code"
    )
    lat: float = pydantic.Field(
        ..., description="Latitude of point within the state or county"
    )
    long: float = pydantic.Field(
        ..., description="Longitude of point within the state or county"
    )
    lastUpdatedDate: datetime.date = pydantic.Field(
        ..., description="Date of latest data"
    )
    projections: Optional[_Projections] = pydantic.Field(...)
    actuals: Optional[_Actuals] = pydantic.Field(...)


# TODO(igor): countyName *must* be None
class CovidActNowStateSummary(CovidActNowAreaSummary):
    stateName: str = pydantic.Field(..., description="The state name")
    countyName: Optional[str] = pydantic.Field(
        default=None, description="The county name"
    )


class CovidActNowCountySummary(CovidActNowAreaSummary):
    stateName: str = pydantic.Field(..., description="The state name")
    countyName: str = pydantic.Field(..., description="The county name")


class CANPredictionTimeseriesRow(pydantic.BaseModel):
    date: datetime.date = pydantic.Field(
        ..., descrition="Date of timeseries data point"
    )
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
    ICUBedCapacity: Optional[int] = pydantic.Field(
        ...,
        description="Number of ICU beds projected to be in-use or actually in use (if in the past)",
    )
    cumulativeDeaths: int = pydantic.Field(..., description="Number of cumulative deaths")
    cumulativeInfected: Optional[int] = pydantic.Field(
        ..., description="Number of cumulative infections"
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
    intervention: str = pydantic.Field(
        ..., description="Name of high-level intervention in-place"
    )
    fips: str = pydantic.Field(
        ..., description="Fips for State + County. Five character code"
    )
    lat: float = pydantic.Field(
        ..., description="Latitude of point within the state or county"
    )
    long: float = pydantic.Field(
        ..., description="Longitude of point within the state or county"
    )
    lastUpdatedDate: datetime.date = pydantic.Field(
        ..., description="Date of latest data"
    )


class CovidActNowStateTimeseries(CovidActNowStateSummary):
    timeseries: List[CANPredictionTimeseriesRow] = pydantic.Field(...)


class CovidActNowCountyTimeseries(CovidActNowCountySummary):
    timeseries: List[CANPredictionTimeseriesRow] = pydantic.Field(...)


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


class CountyFipsSummary(pydantic.BaseModel):
    counties_with_data: List[str] = pydantic.Field(...)
