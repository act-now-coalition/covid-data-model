from typing import Union, Optional, Dict, Any
import enum
from covidactnow.datapublic.common_fields import CommonFields
from libs.datasets import can_model_output_schema as schema


import pandas as pd


class CovidPatientsMethod(enum.Enum):
    ACTUAL = 0
    ESTIMATED = 1


class NonCovidPatientsMethod(enum.Enum):
    ACTUAL = 0
    ESTIMATED_FROM_TYPICAL_UTILIZATION = 1
    ESTIMATED_FROM_TOTAL_ICU_ACTUAL = 2


DEFAULT_ICU_DECOMP = 0.21

ICU_DECOMP_OVERRIDE = {
    "AL": 0.15,
    "AZ": 0.4,
    "DE": 0.3,
    "DC": 0.15,
    "GA": 0.1,
    # TODO(https://trello.com/c/1ddB5ntl/): CCM is currently giving us an
    # extra-high utilization rate. If that gets fixed we may need to bump this
    # back down.
    "MS": 0.37,
    "NV": 0.25,
    "RI": 0,
}


def get_decomp_for_state(state: str) -> float:
    return ICU_DECOMP_OVERRIDE.get(state, DEFAULT_ICU_DECOMP)


class ICUMetricData:
    def __init__(
        self, data: pd.DataFrame, latest_values: Dict[CommonFields, Any], decomp_factor: float
    ):
        self.data = data
        self.latest_values = latest_values
        self.decomp_factor = decomp_factor

    @property
    def actual_current_icu_covid(self) -> Optional[pd.Series]:
        actuals = self.data[CommonFields.CURRENT_ICU]
        if not actuals.any():
            return None

        return actuals

    @property
    def estimated_current_icu_covid(self) -> Optional[pd.Series]:
        estimated = self.data[schema.CURRENT_ICU]
        if not estimated.any():
            return None

        return estimated

    @property
    def actual_current_icu_total(self) -> Optional[pd.Series]:
        actuals = self.data[CommonFields.CURRENT_ICU_TOTAL]
        if not actuals.any():
            return None

        return actuals

    @property
    def total_icu_beds(self) -> Union[pd.Series, float]:
        timeseries = self.data[CommonFields.ICU_BEDS]
        if timeseries.any():
            return timeseries
        return self.latest_values[CommonFields.ICU_BEDS]

    @property
    def typical_usage_rate(self) -> float:
        return self.latest_values[CommonFields.ICU_TYPICAL_OCCUPANCY_RATE]

    @property
    def current_icu_non_covid_with_source(self) -> Tuple[pd.Series, NonCovidPatientsMethod]:
        if self.actual_current_icu_covid is not None and self.actual_current_icu_total is not None:
            source = NonCovidPatientsMethod.ACTUAL
            return (self.actual_current_icu_total - self.actual_current_icu_covid, source)

        if self.actual_current_icu_total is not None:
            source = NonCovidPatientsMethod.ESTIMATED_FROM_TOTAL_ICU_ACTUAL
            return (self.actual_current_icu_total - self.estimated_current_icu_covid, source)

        # Do decomp
        non_covid_utilization = self.typical_usage_rate - self.decomp_factor
        source = NonCovidPatientsMethod.ESTIMATED_FROM_TYPICAL_UTILIZATION

        if isinstance(self.total_icu_beds, pd.Series):
            decomp = non_covid_utilization * self.total_icu_beds
            decomp.name = None
            return (decomp, source)

        total_icu_beds = pd.Series([self.total_icu_beds] * len(self.data), index=self.data.index)
        decomp = non_covid_utilization * total_icu_beds

        return (decomp, source)

    @property
    def current_icu_covid_with_source(self) -> Tuple[pd.Series, CovidPatientsMethod]:
        if self.actual_current_icu_covid is not None:
            source = CovidPatientsMethod.ACTUAL
            return self.actual_current_icu_covid, source

        source = CovidPatientsMethod.ESTIMATED
        return self.estimated_current_icu_covid, source


def calculate_icu_utilization_metric(icu_data: ICUMetricData):
    """

             covid icu patients
           ----------------------
    available beds for covid icu patients


    If we know exactly how many beds are in use by non covid patients we can
    calculate the number of available beds exactly.

    otherwise available beds is calulated by looking at historical data and
    assuming that we can add more capacity.

    So, we need to know or guess how many icu beds are occupied by non covid patients
    non covid patients:
     - get from actuals
     - estimate from typical utilization + decomp factor

    available beds can be calulated

                  covid icu patients
                ----------------------
           total beds - non covid patients
                          ==
                  covid icu patients
                ---------------------
    total beds - (typical non covid usage - decomp)
                          or
                  covid icu patients
                ---------------------
    total beds - (total covid patients - covid patients)


    non covid patient source: Actual total patients |
                              Estimated from actual total patients |
                              Estimated from typical utilization
    covid patient source:     Actuals | Estimates

    """
    non_covid_patients, non_covid_source = icu_data.current_icu_non_covid_with_source
    current_icu_covid, covid_source = icu_data.current_icu_covid_with_source
    metric = current_icu_covid / (icu_data.total_icu_beds - non_covid_patients)

    print(icu_data.total_icu_beds)
    print(non_covid_patients, non_covid_source)

    return {
        "metric": metric,
        "current_icu_covid_source": covid_source,
        "current_icu_non_covid_source": non_covid_source,
    }
