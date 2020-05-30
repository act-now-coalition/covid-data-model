import us
import numpy as np
import pandas as pd
from collections import defaultdict
from epiweeks import Week, Year
from datetime import datetime, timedelta
from pyseir import OUTPUT_DIR, load_data
from pyseir.utils import REF_DATE
from pyseir.load_data import HospitalizationDataType
from statsmodels.nonparametric.kernel_regression import KernelReg
from pyseir.cdc.parameters import Target, ForecastTimeUnit, DATE_FORMAT


def target_column_name(num, target, unit):
    """
    Concatenate number of time units ahead for forecast, name of target and
    the forecast time unit as column name.

    Parameters
    ----------
    num: int
        Number of time units ahead for forecast.
    target: Target
        The target measure.
    unit: ForecastTimeUnit
        The time unit of forecast.

    Yields
    ------
      : str
        generated column name.
    """
    if isinstance(num, int) or isinstance(num, float):
        num = [num]
    for n in num:
        yield f'{int(n)} {unit.value} ahead {target.value}'


def number_of_time_units(ref_date, dates, unit, epi_week=True):
    """
    Generates a sequence of number of time units from given sequence of dates
    from the ref date.
    When unit is ForecastTimeUnit.DAY, return the time gap of given dates
    from the start date in days; when unit is ForecastTimeUnit.WK, return
    time gap from ref date in weeks.
    Note if unit is ForecastTimeUnit.WK, and using epi weeks definition (
    epi_week as True), Saturday is used as the cutoff of a week (ref:
    https://wwwn.cdc.gov/nndss/document/MMWR_Week_overview.pdf).
    In this case, if the ref date is not Saturday, Saturday
    within the same epi week is counted as one epi week from the ref
    date (not zero).

    Parameters
    ----------
    ref_date: datetime.datetime
        Reference date.
    dates: list(datetime.datetime)
        Dates to calculate number of time units since the starting date.
    unit: ForecastTimeUnit
        Time unit of forecast.
    epi_week: bool
        If True, Saturday is used as the cutoff to define the week.
        If ref_date is not Saturday then all following Saturdays are counted
        as its epi week + 1. For example, if ref_date is 2020-05-20,
        then 2020-05-23 is one week from the ref_date and 2020-05-30 is two
        weeks from it, and so on.
        If false, number of weeks from ref date is only determined by
        how many 7 days each date is from the ref date.

    Returns
    -------
    n_units: np.array
        Number of time units from the start date.

    """
    if unit is ForecastTimeUnit.DAY:
        n_units = np.array([(d.date() - ref_date.date()).days for d in dates])
    elif unit is ForecastTimeUnit.WK:
        if epi_week:
            n_units = np.array(
                [Week.fromdate(d).week - Week.fromdate(ref_date).week for d in
                 dates])
            saturdays = np.array([int(d.weekday() == 5) for d in dates])
            # Saturday within same week is counted as next epi week if ref
            # date itself is not Saturday
            n_units += saturdays * int(ref_date.weekday() != 5)
        else:
            n_units = np.array([(d - ref_date).days // 7 for d in dates])

    return n_units


def smooth_timeseries(time_steps, data):
    """
    Smoothing time series data.

    Parameters
    ----------
    time_steps: list or np.array
        Time steps from some starting date.
    data: list or np.array
        data to smooth

    Returns
    -------
    smoothed: np.array
        Data smoothed through given time steps.
    """

    kr = KernelReg(data, time_steps, 'c')
    smoothed, _ = kr.fit(time_steps)
    return smoothed


def aggregate_timeseries(dates, data, unit, target):
    """
    Aggregates time series data based on time unit and forecast target.
    There is no aggregation if unit is ForecastTimeUnit.DAY. If unit is
    ForecastTimeUnit.WK, do aggregation based on type of forecast target:
    if target is incident death or incident hospitalizations, the time series
    data is aggregated as weekly sum; if target is cumulative death,
    the time series is aggregated as the value on Saturdays.

    Parameters
    ----------
    dates: list or np.array
        Dates of the time series.
    data: list or np.array
        Time series data
    unit: ForecastTimeUnit
        Time unit to aggregate the data
    target: Target
        Forecast target, cumulative death, incident death and incident
        hospitalizations

    Returns
    -------
    agg_dates: np.array
        Date corresponding to the aggregated time series data.
    agg_data: np.array
        Aggregated time series data.
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
            agg_dates = np.intersect1d(dates, saturdays)
            agg_data = data[np.searchsorted(dates, agg_dates)]

    else:
        raise ValueError(f'{unit} is not implemented')

    return agg_dates, agg_data


def load_and_aggregate_observations(fips, units, targets, end_date=None, smooth=True):
    """
    Load observations based on type of target (cumulative death, incident
    death and incident hospitalizations) and unit (day, week).

    Parameters
    ----------
    fips: str
        Two digits State FIPS code.
    units: list
        List of ForecastTimeUnit objects.
    targets: list
        List of Target objects.
    end_date: datetime.datetime
        End date of the observations. Default None, if set observation after
        this date will be removed.

    Returns
    -------
      : dict(dict)
        Contains observed cumulative deaths, incident deaths,
        and incident hospitalizations, with target name as primary key and
        forecast time unit as secondary key, and corresponding time series of
        observations as values:
        <target>:
            <forecast time unit>: pd.Series
                With date string as index and observations as values.
        Observations for hospitalizations can be None if no
        cumulative hospitalization data is available for the FIPS code.
    """

    times, observed_new_cases, observed_new_deaths = \
        load_data.load_new_case_data_by_state(us.states.lookup(fips).name,
                                              REF_DATE)
    times = np.array(times)
    hospital_times, hospitalizations, hospitalization_data_type = \
        load_data.load_hospitalization_data_by_state(us.states.lookup(fips).abbr,
                                                     REF_DATE)
    hospital_times = np.array(hospital_times)
    # sometimes we may want to check the performance of historical forecast,
    # here it enables blocking part of observations.
    end_date = end_date or datetime.today()
    maximum_time_step = (end_date.date() - REF_DATE.date()).days

    observed_new_cases = observed_new_cases[times <= maximum_time_step]
    observed_new_deaths = observed_new_deaths[times <= maximum_time_step]
    times = times[times <= maximum_time_step]

    if hospital_times is not None:
        hospitalizations = hospitalizations[hospital_times <= maximum_time_step]
        hospital_times = hospital_times[hospital_times <= maximum_time_step]

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
                agg_dates, agg_observations = aggregate_timeseries(observation_dates[target.value],
                                                                   raw_observations[target.value],
                                                                   unit,
                                                                   target)

                # smoothing observed daily incident deaths or hospitalizations
                if smooth:
                    if target in [Target.INC_DEATH, Target.INC_HOSP]:
                        if unit is ForecastTimeUnit.DAY:
                            agg_observations = smooth_timeseries([(d - REF_DATE).days for d in agg_dates],
                                                                  agg_observations).clip(min=0)

                observations[target.value][unit.value] = \
                    pd.Series(agg_observations,
                              index=pd.DatetimeIndex(agg_dates).strftime(DATE_FORMAT))

            else:
                observations[target.value][unit.value] = None

    return dict(observations)
