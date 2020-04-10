from typing import List
import pydantic

class _CountyCases(pydantic.BaseModel):
    fips: str = pydantic.Field(..., description="FIPS state code + FIPS county code")
    cases: int = pydantic.Field(..., description="Cumulative case count.")
    deaths: int = pydantic.Field(..., description="Cumulative deaths count.")
    date: str = pydantic.Field(..., description="Date of latest data")



class StateCaseSummary(pydantic.BaseModel):
    """Case summary output in format that website expects for embeds."""

    state: str = pydantic.Field(..., description="2 letter state code")
    date: str = pydantic.Field(..., description="Date of latest data")
    cases: int = pydantic.Field(..., description="Cumulative case count.")
    deaths: int = pydantic.Field(..., description="Cumulative deaths count.")
    counties: List[_CountyCases] = pydantic.Field(...)
