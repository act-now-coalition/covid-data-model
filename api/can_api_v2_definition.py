from typing import List, Optional
from libs.enums import Intervention
from libs.datasets.dataset_utils import AggregationLevel
from api import can_api_definition
from libs import us_state_abbrev
from libs import base_model
import pydantic
import datetime


class HospitalResourceUtilization(base_model.APIBaseModel):
    capacity: int = pydantic.Field(None, description="Total capacity for resource.")
    currentUsageTotal: int = pydantic.Field(
        None, description="Currently used capacity for resource by all patients (COVID + Non-COVID)"
    )
    currentUsageCovid: int = pydantic.Field(
        None, description="Currently used capacity for resource by COVID "
    )
    typicalUsageRate: float = pydantic.Field(
        None, description="Typical used capacity rate for resource. This excludes any COVID usage."
    )


class Actuals(base_model.APIBaseModel):
    """Known actuals data."""

    cases: int = pydantic.Field(
        None, description="Cumulative number of confirmed or suspected cases"
    )
    deaths: int = pydantic.Field(
        None,
        description=(
            "Cumulative number of deaths that are suspected or "
            "confirmed to have been caused by COVID-19"
        ),
    )
    positiveTests: int = pydantic.Field(None)
    negativeTests: int = pydantic.Field(None)
    contactTracers: int = pydantic.Field(None, description="# of Contact Tracers")
    hospitalBeds: HospitalResourceUtilization = pydantic.Field(None)
    icuBeds: HospitalResourceUtilization = pydantic.Field(None)


class ActualsTimeseriesRow(Actuals):
    """Actual data for a specific day."""

    date: datetime.date = pydantic.Field(..., descrition="Date of timeseries data point")


class Metrics(base_model.APIBaseModel):
    """Calculated metrics data based on known actuals."""

    testPositivityRatio: float = pydantic.Field(
        None,
        description="Ratio of people who test positive calculated using a 7-day rolling average.",
    )

    caseDensity: float = pydantic.Field(
        None,
        description="The number of cases per 100k population calculated using a 7-day rolling average.",
    )

    contactTracerCapacityRatio: float = pydantic.Field(
        None,
        description=(
            "Ratio of currently hired tracers to estimated "
            "tracers needed based on 7-day daily case average."
        ),
    )

    infectionRate: float = pydantic.Field(
        None, description="R_t, or the estimated number of infections arising from a typical case."
    )

    infectionRateCI90: float = pydantic.Field(
        None,
        description="90th percentile confidence interval upper endpoint of the infection rate.",
    )
    icuHeadroomRatio: float = pydantic.Field(None)
    icuHeadroomDetails: can_api_definition.ICUHeadroomMetricDetails = pydantic.Field(None)


class MetricsTimeseriesRow(Metrics):
    """Metrics data for a specific day."""

    date: datetime.date = pydantic.Field(..., descrition="Date of timeseries data point")


class RegionSummary(base_model.APIBaseModel):
    """Summary of actual and prediction data for a single region."""

    fips: str = pydantic.Field(
        ...,
        description="Fips Code.  For state level data, 2 characters, for county level data, 5 characters.",
    )
    country: str = pydantic.Field(..., description="2-letter ISO-3166 Country code.")
    state: str = pydantic.Field(..., description="2-letter ANSI state code.")
    county: str = pydantic.Field(None, description="County name")

    level: AggregationLevel = pydantic.Field(..., description="Level of region.")
    lat: float = pydantic.Field(None, description="Latitude of point within the state or county")
    long: float = pydantic.Field(None, description="Longitude of point within the state or county")
    population: int = pydantic.Field(
        ..., description="Total Population in geographic region.", gt=0
    )

    metrics: Metrics = pydantic.Field(None)
    actuals: Actuals = pydantic.Field(None)

    lastUpdatedDate: datetime.date = pydantic.Field(..., description="Date of latest data")


class RegionSummaryWithTimeseries(RegionSummary):
    """Summary data for a region with prediction timeseries data and actual timeseries data."""

    metricsTimeseries: List[MetricsTimeseriesRow] = pydantic.Field(None)
    actualsTimeseries: List[ActualsTimeseriesRow] = pydantic.Field(...)

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

    @property
    def level(self) -> AggregationLevel:
        return self.__root__[0].level


class AggregateRegionSummaryWithTimeseries(base_model.APIBaseModel):
    """Timeseries and summary data for multiple regions."""

    __root__: List[RegionSummaryWithTimeseries] = pydantic.Field(...)

    @property
    def level(self) -> AggregationLevel:
        return self.__root__[0].level


class MetricsTimeseriesRowWithHeader(MetricsTimeseriesRow):
    """Prediction timeseries row with location information."""

    country: str = "US"
    state: str = pydantic.Field(..., description="The state name")
    county: str = pydantic.Field(None, description="The county name")
    fips: str = pydantic.Field(..., description="Fips for State + County. Five character code")
    lat: float = pydantic.Field(None, description="Latitude of point within the state or county")
    long: float = pydantic.Field(None, description="Longitude of point within the state or county")
    lastUpdatedDate: datetime.date = pydantic.Field(..., description="Date of latest data")

    @property
    def aggregate_level(self) -> AggregationLevel:
        if len(self.fips) == 2:
            return AggregationLevel.STATE

        if len(self.fips) == 5:
            return AggregationLevel.COUNTY


class AggregateFlattenedTimeseries(base_model.APIBaseModel):
    """Flattened prediction timeseries data for multiple regions."""

    __root__: List[MetricsTimeseriesRowWithHeader] = pydantic.Field(...)

    @property
    def aggregate_level(self) -> AggregationLevel:
        return self.__root__[0].aggregate_level
