from typing import List
import pydantic

"""
Documented https://www.dropbox.com/scl/fi/o4bec2kz8dkcxdtabtqda/CAN-V1-API-Draft.paper?dl=0&rlkey=f7elx3zmo6tt5s7mj5nvap9a0
Eventually to become: '
{
  stateName,
  countyName,
  fips,
  lat, 
  long,
  lastUpdatedDate, // ISO 8601 date string
  actuals: {
    population,
    intervention, // current intervention (limited_action, social_distancing, stay_at_home)
    cumulativeConfirmedCases,
    cumulativeDeaths,
    hospitalBeds: {
      capacity,
      currentUsage,
    }, 
    ICUBeds: { same as above }   
  }, 
  projections: {
    intervention, // name of intervention these projections reflect
    peakDeaths,
    peakDeathsDate,
    hospitalBeds: {
      hospitalOverloadDate, // null if no shortage projected
      peakInUse,
      peakUseDate,
      peakShortfall,
    },
    ICUBeds: { same as above },
  },
  timeseries: [{
    date,
    hospitalBedsInUse,
    hospitalBedCapacity,
    ICUBedsInUse,
    ICUBedCapacity,
    newDeaths,
    newConfirmedCases,
    estimatedNewInfections,
    isProjected,
  }],
};
"""

class _HospitalBeds(pydantic.BaseModel):
    peakNeeded: int = pydantic.Field(..., description="Beds Needed at the peak hospitalizaitons")
    peakShortfall: int = pydantic.Field(..., description="Shortfall of beds needed at the peek hospitalizaitons")
    peakDate: str = pydantic.Field(..., description="Date of peak hospitalizations")
    shortageStartDate: str = pydantic.Field(..., description="Date when hospitals overload")


class _Projections(pydantic.BaseModel):
    hospitalBeds: _HospitalBeds = pydantic.Field(...)


class CANPredictionAPI(pydantic.BaseModel):
    stateName: str = pydantic.Field(..., description="The state name")
    countyName: str = pydantic.Field(..., description="The county name")
    fips: str = pydantic.Field(..., description="Fips for State + County. Five character code")
    lastUpdatedDate: str = pydantic.Field(..., description="Date of latest data")
    projections: _Projections = pydantic.Field(...)