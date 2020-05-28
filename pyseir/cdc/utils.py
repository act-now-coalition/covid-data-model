from enum import Enum
import us
import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import timedelta, date
from pyseir import OUTPUT_DIR, load_data
from pyseir.utils import REF_DATE
from pyseir.load_data import HospitalizationDataType
from statsmodels.nonparametric.kernel_regression import KernelReg


DATE_FORMAT = '%Y-%m-%d'

class Target(Enum):
    CUM_DEATH = 'cum death'
    INC_DEATH = 'inc death'
    CUM_HOSP = 'cum hosp'
    INC_HOSP = 'inc hosp'


class ForecastTimeUnit(Enum):
    DAY = 'day'
    WK = 'wk'


class ForecastUncertainty(Enum):
    DEFAULT = 'default'
    NAIVE = 'naive'


def target_column_name(num, target, time_unit):
    """
    Concatenate number of time units ahead for forecast, name of target and
    the forecast time unit as column name.

    Parameters
    ----------
    num: int
        Number of time units ahead for forecast.
    target: Target
        The target measure.
    time_unit: ForecastTimeUnit
        The time unit of forecast.

    Yields
    ------
      : str
        generated column name.
    """
    num = list(num)
    for n in num:
        yield f'{int(n)} {time_unit.value} ahead {target.value}'


def number_of_units(start_date, dates, unit):
    """

    """
    if unit is ForecastTimeUnit.DAY:
        num_days = 1
    elif unit is ForecastTimeUnit.WK:
        num_days = 7
    n_units = [(d - start_date).days // num_days + 1 for d in dates]
    return n_units

def smooth_observations(time, observations):
    """

    """
    kr = KernelReg(observations, time, 'c')
    smoothed, _ = kr.fit(time)
    return smoothed


def aggregate_observations(dates, data, unit, target):
    """

    """
    dates = np.array(dates)
    data = np.array(data)
    if unit is ForecastTimeUnit.DAY:
        agg_dates = dates
        agg_data = data

    elif unit is ForecastTimeUnit.WK:
        saturdays = np.array([d + timedelta((12 - d.weekday()) % 7) for d in dates])
        data = data[saturdays <= dates.max()]
        saturdays = saturdays[saturdays <= dates.max()]
        if target in [Target.INC_HOSP, Target.INC_DEATH]:
            agg = list()
            for d in np.unique(saturdays):
                agg.append(data[saturdays == d].sum())
            agg_data = np.array(agg)
            agg_dates = np.unique(saturdays)
        else:
            agg_data = data[dates == saturdays]
            agg_dates = np.unique(np.intersect1d(dates, saturdays))

    else:
        raise ValueError(f'{unit} is not implemented')

    return agg_dates.flatten(), agg_data.flatten()


def load_and_aggregate_observations(fips, units, targets, smooth=True):
    """
    Load observations based on type of target and unit.

    Returns
    -------

    """

    times, observed_new_cases, observed_new_deaths = \
        load_data.load_new_case_data_by_state(us.states.lookup(fips).name,
                                              REF_DATE)

    hospital_times, hospitalizations, hospitalization_data_type = \
        load_data.load_hospitalization_data_by_state(us.states.lookup(fips).abbr,
                                                     REF_DATE)

    observation_dates = {}
    for target in [Target.CUM_DEATH, Target.INC_DEATH]:
        observation_dates[target.value] = [timedelta(int(t)) + REF_DATE for t in times]
    raw_observations = {Target.CUM_DEATH.value: observed_new_deaths.cumsum(),
                        Target.INC_DEATH.value: observed_new_deaths}

    if hospital_times is not None:
        hospital_dates = [timedelta(int(t)) + REF_DATE for t in hospital_times]
        if hospitalization_data_type is HospitalizationDataType.CUMULATIVE_HOSPITALIZATIONS:
            raw_observations[Target.INC_HOSP.value] = np.append(hospitalizations[0],
                                                                np.diff(hospitalizations))
            observation_dates[Target.INC_HOSP.value] = hospital_dates

    observations = defaultdict(dict)

    for unit in units:
        for target in targets:
            if target.value in raw_observations:

                agg_dates, agg_observations = aggregate_observations(observation_dates[target.value],
                                                                     raw_observations[target.value],
                                                                     unit,
                                                                     target)

                # smoothing observed daily incident deaths or hospitalizations
                if smooth:
                    if target in [Target.INC_DEATH, Target.INC_HOSP]:
                        if unit is ForecastTimeUnit.DAY:
                            agg_observations = smooth_observations([(d - REF_DATE).days for d in agg_dates],
                                                                    agg_observations).clip(min=0)

                observations[target.value][unit.value] = \
                    pd.Series(agg_observations,
                              index=pd.DatetimeIndex(agg_dates).strftime(DATE_FORMAT))

            else:
                observations[target.value][unit.value] = None

    return observations

