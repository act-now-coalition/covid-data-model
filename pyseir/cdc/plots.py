import logging
import os
import us
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
from datetime import datetime
from matplotlib.backends import backend_pdf
from pyseir import load_data
from pyseir.cdc.parameters import Target, ForecastTimeUnit, TARGETS_AND_UNITS_REPORT
from pyseir.cdc.utils import load_and_aggregate_observations
from pyseir.cdc.output_mapper import (OutputMapper, REPORT_FOLDER, TEAM,
                                      MODEL, FORECAST_DATE, DATE_FORMAT)


LABELS = {Target.CUM_DEATH: 'cumulative death',
          Target.INC_DEATH: 'incident death',
          Target.INC_CASE: 'incident case'}


def load_output_mapper_result(fips, forecast_date):
    """

    """

    if isinstance(forecast_date, datetime):
        forecast_date = datetime.strftime(forecast_date, DATE_FORMAT)

    om_result = pd.read_csv(f'{REPORT_FOLDER}/{forecast_date}/'
                            f'{forecast_date}_{TEAM}_{MODEL}_{fips}.csv')
    return om_result


def plot_results(fips, om_result, targets_and_units, observations, pdf=None):
    """
    Plot median and 95% confidence interval of the forecast and observations
    for a given type of target (cumulative death, incident death or incident
    hospitalizations).

    Parameters
    ----------
    fips: str
        State FIPS code.
    forecast_date: datetime.datetime or str
        date when the forecast is made
    target: Target
        Target of forecast, cumulative death, incident death or
        incident hospitalizations.
    observations: dict(dict)
        Contains observed cumulative deaths, incident deaths,
        and incident hospitalizations, with target name as primary key and
        forecast time unit as secondary key, and corresponding time series of
        observations as values:
        <target>:
            <forecast time unit>: pd.Series
                With date string as index and observations as values.
        Observations for hospitalizations can be None if no
        cumulative hospitalization data is available for the FIPS code.
    pdf: matplotlib.backends.backend_pdf.PdfPages
        Pdf object to save the figures.
    """

    results = defaultdict(dict)

    for target, unit in targets_and_units:
        df = om_result[om_result.target.str.contains(f'{unit.value} ahead {target.value}')].copy()
        results[target.value][unit.value] = list()
        for target_name in df.target.unique():
            sub_df = df[df.target == target_name]
            sub_df['quantile'] = sub_df['quantile'].apply(lambda x: float(x))

            results[target.value][unit.value].append(pd.DataFrame({
                'target_end_date': [sub_df.iloc[0]['target_end_date']],
                'median': [float(sub_df[sub_df['quantile'] == 0.5]['value'])],
                'ci_250': [float(sub_df[sub_df['quantile'] == 0.25]['value'])],
                'ci_750': [float(sub_df[sub_df['quantile'] == 0.75]['value'])],
                'ci_025': [float(sub_df[sub_df['quantile'] == 0.025]['value'])],
                'ci_975': [float(sub_df[sub_df['quantile'] == 0.975]['value'])]}))

    for target_name in results:
        for unit_name in results[target_name]:
            results[target_name][unit_name] = pd.concat(results[target_name][unit_name])
            results[target_name][unit_name]['target_end_date'] = \
                pd.to_datetime(results[target_name][unit_name]['target_end_date'])

    plt.figure(figsize=(10, 10))
    for n, target_unit in enumerate(targets_and_units):
        target, unit = target_unit
        plt.subplot(2, 2, n+1)
        plt.plot(pd.to_datetime(observations[target.value][unit.value].index),
                 observations[target.value][unit.value].values,
                 label='observed')

        plt.plot(results[target.value][unit.value]['target_end_date'],
                 results[target.value][unit.value]['median'],
                 marker='o', label='forecast median')
        plt.fill_between(x=results[target.value][unit.value]['target_end_date'],
                         y1=results[target.value][unit.value]['ci_025'],
                         y2=results[target.value][unit.value]['ci_975'],
                         label=f'CI_95',
                         alpha=0.15)

        plt.fill_between(x=results[target.value][unit.value]['target_end_date'],
                         y1=results[target.value][unit.value]['ci_250'],
                         y2=results[target.value][unit.value]['ci_750'],
                         label=f'CI_50',
                         alpha=0.3)
        plt.ylabel(f'{unit} ahead forecast')
        plt.xlabel('time')
        plt.legend()
        plt.xticks(rotation=45)

        plt.title('/'.join([us.states.lookup(fips).name, fips, LABELS[target]]))

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)

    if pdf:
        pdf.savefig()


def run_all(forecast_date=FORECAST_DATE):
    """
    Plot confidence intervals of forecast targests for States and save
    figures to pdf.

    Parameters
    ----------
    forecast_date: datetime.datetime
        Date of the forecast.
    """

    df_whitelist = load_data.load_whitelist()
    df_whitelist = df_whitelist[df_whitelist['inference_ok'] == True]
    fips_list = list(df_whitelist['fips'].str[:2].unique())
    date_string = datetime.strftime(forecast_date, DATE_FORMAT)
    output_path = os.path.join(f'{REPORT_FOLDER}',
                               date_string,
                               f'report_{date_string}.pdf')
    pdf = backend_pdf.PdfPages(output_path)
    for fips in fips_list:
        logging.info(f'plotting cdc submission for fips {fips}')
        observations = load_and_aggregate_observations(fips,
                                                       targets_and_units=[(Target(tup[0]),
                                                                           ForecastTimeUnit(tup[1])) for
                                                                          tup in TARGETS_AND_UNITS_REPORT],
                                                       smooth=False)
        om_result = load_output_mapper_result(fips, datetime.strftime(forecast_date, DATE_FORMAT))
        plot_results(fips=fips, om_result=om_result,
                     observations=observations,
                     pdf=pdf,
                     targets_and_units=[(Target(tup[0]), ForecastTimeUnit(tup[1]))
                                        for tup in TARGETS_AND_UNITS_REPORT if
                                        Target(tup[0]) != Target.INC_HOSP])

    pdf.close()
