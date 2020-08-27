from typing import Union, Optional, Dict, Any, Tuple
import enum
import numpy as np
from covidactnow.datapublic.common_fields import CommonFields
from libs.datasets import can_model_output_schema as schema
from libs import series_utils
from api import can_api_definition

import pandas as pd


NonCovidPatientsMethod = can_api_definition.NonCovidPatientsMethod
CovidPatientsMethod = can_api_definition.CovidPatientsMethod


# Default utilization to use (before decomp) if there isn't a location-specific
# value in the API.
DEFAULT_ICU_UTILIZATION = 0.75

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
        self,
        timeseries_data: pd.DataFrame,
        estimated_current_icu: Optional[pd.Series],
        latest_values: Dict[CommonFields, Any],
        decomp_factor: float,
        require_recent_data: bool = True,
    ):
        self._data = timeseries_data
        self._estimated_current_icu = estimated_current_icu
        self._latest_values = latest_values
        self.decomp_factor = decomp_factor
        self._require_recent_data = require_recent_data

    @property
    def actual_current_icu_covid(self) -> Optional[pd.Series]:
        actuals = self._data[CommonFields.CURRENT_ICU]
        if not actuals.any():
            return None

        if self._require_recent_data and not series_utils.has_recent_data(actuals):
            return None

        return actuals

    @property
    def actual_current_icu_total(self) -> Optional[pd.Series]:
        actuals = self._data[CommonFields.CURRENT_ICU_TOTAL]
        if not actuals.any():
            return None

        if self._require_recent_data and not series_utils.has_recent_data(actuals):
            return None

        return actuals

    @property
    def estimated_current_icu_covid(self) -> Optional[pd.Series]:
        estimated = self._estimated_current_icu
        if estimated is None or not estimated.any():
            return None

        return estimated

    @property
    def total_icu_beds(self) -> Union[pd.Series, float]:
        timeseries = self._data[CommonFields.ICU_BEDS]
        if timeseries.any():
            return timeseries
        return self._latest_values[CommonFields.ICU_BEDS]

    @property
    def typical_usage_rate(self) -> float:
        typical_occupancy = self._latest_values[CommonFields.ICU_TYPICAL_OCCUPANCY_RATE]
        if typical_occupancy is None or np.isnan(typical_occupancy):
            return DEFAULT_ICU_UTILIZATION

        return typical_occupancy

    @property
    def current_icu_non_covid_with_source(self) -> Tuple[pd.Series, NonCovidPatientsMethod]:
        """Returns non-covid ICU patients and method used in calculation."""
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

        total_icu_beds = pd.Series([self.total_icu_beds] * len(self._data), index=self._data.index)
        decomp = non_covid_utilization * total_icu_beds

        return (decomp, source)

    @property
    def current_icu_covid_with_source(self) -> Tuple[pd.Series, CovidPatientsMethod]:
        if self.actual_current_icu_covid is not None:
            source = CovidPatientsMethod.ACTUAL
            return self.actual_current_icu_covid, source

        source = CovidPatientsMethod.ESTIMATED
        return self.estimated_current_icu_covid, source


def calculate_icu_utilization_metric(
    icu_data: ICUMetricData,
) -> Tuple[Optional[pd.Series], Optional[can_api_definition.ICUHeadroomMetricDetails]]:
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
    current_icu_covid, covid_source = icu_data.current_icu_covid_with_source
    if current_icu_covid is None:
        return np.nan, None

    non_covid_patients, non_covid_source = icu_data.current_icu_non_covid_with_source
    metric = current_icu_covid / (icu_data.total_icu_beds - non_covid_patients)

    details = can_api_definition.ICUHeadroomMetricDetails(
        currentIcuCovidMethod=covid_source, currentIcuNonCovidMethod=non_covid_source
    )
    return metric, details
