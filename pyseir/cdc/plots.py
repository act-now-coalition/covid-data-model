import us
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
from datetime import datetime
from matplotlib.backends import backend_pdf
from pyseir import load_data
from pyseir.cdc.parameters import Target, ForecastTimeUnit
from pyseir.cdc.utils import load_and_aggregate_observations
from pyseir.cdc.output_mapper import (OutputMapper, REPORT_FOLDER, TEAM,
                                      MODEL, FORECAST_DATE, DATE_FORMAT)


LABELS = {Target.CUM_DEATH: 'cumulative death',
          Target.INC_DEATH: 'incident death'}



def plot_results(fips, forecast_date, target, observations, pdf=None):
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

    if isinstance(forecast_date, datetime):
        forecast_date = datetime.strftime(forecast_date, DATE_FORMAT)

    df = pd.read_csv(f'{REPORT_FOLDER}/{forecast_date}_{TEAM}_{MODEL}_{fips}.csv')

    results = defaultdict(list)

    for unit in ['day', 'wk']:
        df_by_unit = df[df.target.str.contains(f'{unit} ahead {target.value}')].copy()
        for target_name in df_by_unit.target.unique():
            sub_df = df_by_unit[df_by_unit.target == target_name]
            sub_df['quantile'] = sub_df['quantile'].apply(lambda x: float(x))

            results[unit].append(pd.DataFrame({
                'target_end_date': [sub_df.iloc[0]['target_end_date']],
                'median': [float(sub_df[sub_df['quantile'] == 0.5]['value'])],
                'ci_250': [float(sub_df[sub_df['quantile'] == 0.25]['value'])],
                'ci_750': [float(sub_df[sub_df['quantile'] == 0.75]['value'])],
                'ci_025': [float(sub_df[sub_df['quantile'] == 0.025]['value'])],
                'ci_975': [float(sub_df[sub_df['quantile'] == 0.975]['value'])]}))

    for unit in [ForecastTimeUnit.DAY.value, ForecastTimeUnit.WK.value]:
        results[unit] = pd.concat(results[unit])
        results[unit]['target_end_date'] = pd.to_datetime(results[unit]['target_end_date'])

    plt.figure(figsize=(10, 6))
    for n, unit in enumerate([ForecastTimeUnit.DAY.value,
                              ForecastTimeUnit.WK.value]):
        plt.subplot(1, 2, n+1)
        plt.plot(pd.to_datetime(observations[target.value][unit].index),
                 observations[target.value][unit].values,
                 label='observed')

        plt.plot(results[unit]['target_end_date'], results[unit]['median'],
                 marker='o', label='forecast median')
        plt.fill_between(x=results[unit]['target_end_date'],
                         y1=results[unit]['ci_025'],
                         y2=results[unit]['ci_975'],
                         label=f'CI_95',
                         alpha=0.15)
        plt.fill_between(x=results[unit]['target_end_date'],
                         y1=results[unit]['ci_250'],
                         y2=results[unit]['ci_750'],
                         label=f'CI_50',
                         alpha=0.3)
        plt.ylabel(f'{unit} ahead forecast')
        plt.xlabel('time')
        plt.legend()
        plt.xticks(rotation=45)

    plt.suptitle('/'.join([us.states.lookup(fips).name, fips, LABELS[target]]))
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
    output_path = f'{REPORT_FOLDER}/report_{datetime.strftime(forecast_date, DATE_FORMAT)}.pdf'
    pdf = backend_pdf.PdfPages(output_path)
    for fips in fips_list:
        observations = load_and_aggregate_observations(fips,
                                                       units=[ForecastTimeUnit.DAY, ForecastTimeUnit.WK],
                                                       targets=[Target.CUM_DEATH, Target.INC_DEATH],
                                                       smooth=False)
        for target in [Target.CUM_DEATH, Target.INC_DEATH]:
            plot_results(fips, datetime.strftime(forecast_date, DATE_FORMAT),
                         target, observations, pdf)
    pdf.close()
