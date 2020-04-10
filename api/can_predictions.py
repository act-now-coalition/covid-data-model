from typing import List
import pydantic
import datetime

"""
Documented https://www.dropbox.com/scl/fi/o4bec2kz8dkcxdtabtqda/CAN-V1-API-Draft.paper?dl=0&rlkey=f7elx3zmo6tt5s7mj5nvap9a0

"""

class _HospitalBeds(pydantic.BaseModel):
    peakShortfall: int = pydantic.Field(default=0, description="Shortfall of beds needed at the peek hospitalizaitons")
    peakDate: datetime.datetime = pydantic.Field(default='', description="Date of peak hospitalizations")
    shortageStartDate: datetime.datetime = pydantic.Field(default='', description="Date when hospitals overload")


class _Projections(pydantic.BaseModel):
    aggregateDeaths: int = pydantic.Field(...)
    hospitalBeds: _HospitalBeds = pydantic.Field(...)

class CANPredictionAPIRow(pydantic.BaseModel):
    stateName: str = pydantic.Field(..., description="The state name")
    countyName: str = pydantic.Field(..., description="The county name")
    fips: str = pydantic.Field(..., description="Fips for State + County. Five character code")
    lastUpdatedDate: str = pydantic.Field(..., description="Date of latest data")
    projections: _Projections = pydantic.Field(...)

class CANPredictionAPI(pydantic.BaseModel):
    data: List[CANPredictionAPIRow] = pydantic.Field(...)