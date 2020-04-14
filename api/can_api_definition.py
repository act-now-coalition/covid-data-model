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
    totalHospitalBeds: _ResourceUsageProjection = pydantic.Field(..., description="Projection about total hospital bed utilization")
    ICUBeds: Optional[_ResourceUsageProjection] = pydantic.Field(..., description="Projection about ICU hospital bed utilization")
    cumulativeDeaths: int =  pydantic.Field(..., description="Cumulative number of deaths within projection window")
    # TODO(igor): make non-optional
    peakDeaths: Optional[int] = pydantic.Field(..., description="Peak number of deaths within projection window")
    peakDeathsDate: Optional[datetime.date] = pydantic.Field(..., description="Date of peak number of deaths")
    endDate: datetime.date = pydantic.Field(..., description="Date the projection goes until")

class _ResourceUtilization(pydantic.BaseModel):
    capacity: int = pydantic.Field(..., description="Total capacity for resource")
    currentUsage: int = pydantic.Field(..., description="Currently used capacity for resource")

class _Actuals(pydantic.BaseModel):
    population: int = pydantic.Field(..., description="Total population in geographic area")
    intervention: str = pydantic.Field(..., description="Name of high-level intervention in-place")
    cumulativeConfirmedCases: int = pydantic.Field(..., description="Number of confirmed cases so far")
    cumulativeDeaths: int = pydantic.Field(..., description="Number of deaths so far")
    hospitalBeds: _ResourceUtilization = pydantic.Field(...)
    ICUBeds: Optional[_ResourceUtilization] = pydantic.Field(...)

class CovidActNowAreaSummary(pydantic.BaseModel):
    countryName: str = 'US'
    fips: str = pydantic.Field(
        ..., description="Fips for State + County. Five character code"
    )
    lat: float = pydantic.Field(..., description="Lattitude of point within the state or county")
    long: float = pydantic.Field(..., description="Longitude of point within the state or county")
    lastUpdatedDate: Optional[datetime.date] = pydantic.Field(..., description="Date of latest data")
    projections: Optional[_Projections] = pydantic.Field(...)
    actuals: Optional[_Actuals] = pydantic.Field(...)

# TODO(igor): countyName *must* be None
class CovidActNowStateSummary(CovidActNowAreaSummary):
    stateName: str = pydantic.Field(..., description="The state name")
    countyName: Optional[str] = pydantic.Field(default=None, description="The county name")

class CovidActNowCountySummary(CovidActNowAreaSummary):
    stateName: str = pydantic.Field(..., description="The state name")
    countyName: str = pydantic.Field(..., description="The county name")

class CANPredictionTimeseriesRow(pydantic.BaseModel):
    date: datetime.date = pydantic.Field(..., descrition="Date of timeseries data point")
    hospitalBedsInUse: int = pydantic.Field(..., description="Number of hospital beds projected to be in-use or that were actually in use (if in the past)")
    hospitalBedCapacity: int = pydantic.Field(..., description="Number of hospital beds projected to be in-use or actually in use (if in the past)")
    ICUBedsInUse: int = pydantic.Field(..., description="Number of ICU beds projected to be in-use or that were actually in use (if in the past)")
    ICUBedCapacity: int = pydantic.Field(..., description="Number of ICU beds projected to be in-use or actually in use (if in the past)")
    newDeaths: int = pydantic.Field(..., description="Number of new deaths")
    newConfirmedCases: int = pydantic.Field(..., description="Number of new confirmed cases")
    newInfections: int = pydantic.Field(..., description="Number of new infections")

class CovidActNowStateTimeseries(CovidActNowStateSummary):
    timeseries: List[CANPredictionTimeseriesRow] = pydantic.Field(...)

class CovidActNowCountyTimeseries(CovidActNowCountySummary):
    timeseries: List[CANPredictionTimeseriesRow] = pydantic.Field(...)

class CovidActNowCountiesAPI(pydantic.BaseModel):
    data: List[CovidActNowCountySummary] = pydantic.Field(...)

class CovidActNowStatesAPI(pydantic.BaseModel):
    data: List[CovidActNowAreaSummary] = pydantic.Field(...)
