from typing import List, Optional
import pydantic
import datetime

"""
Documented https://www.dropbox.com/scl/fi/o4bec2kz8dkcxdtabtqda/CAN-V1-API-Draft.paper?dl=0&rlkey=f7elx3zmo6tt5s7mj5nvap9a0

"""


class _HospitalBeds(pydantic.BaseModel):
    peakShortfall: Optional[int] = pydantic.Field(
        ..., description="Shortfall of beds needed at the peek hospitalizaitons"
    )
    peakDate: Optional[datetime.datetime] = pydantic.Field(
        ..., description="Date of peak hospitalizations"
    )
    shortageStartDate: Optional[datetime.datetime] = pydantic.Field(
        ..., description="Date when hospitals overload"
    )


class _Projections(pydantic.BaseModel):
    hospitalBeds: _HospitalBeds = pydantic.Field(...)


class CANPredictionAPIRow(pydantic.BaseModel):
    stateName: str = pydantic.Field(..., description="The state name")
    countyName: str = pydantic.Field(..., description="The county name")
    fips: str = pydantic.Field(
        ..., description="Fips for State + County. Five character code"
    )
    lastUpdatedDate: str = pydantic.Field(..., description="Date of latest data")
    projections: _Projections = pydantic.Field(...)


class CANPredictionAPI(pydantic.BaseModel):
    data: List[CANPredictionAPIRow] = pydantic.Field(...)
