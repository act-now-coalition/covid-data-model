from pyseir.inference.fit_results import load_mle_model, load_inference_result
from pyseir import load_data
import pandas as pd
from pyseir.utils import REF_DATE
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.backends import backend_pdf
import us
import matplotlib
from pyseir.cdc.utils import Target, ForecastTimeUnit, aggregate_observations, load_and_aggregate_observations
from pyseir.cdc.output_mapper import OutputMapper, REPORT_FOLDER, TEAM, MODEL, FORECAST_DATE, DATE_FORMAT
from collections import defaultdict
from epiweeks import Week, Year

LABELS = {Target.CUM_DEATH: 'cumulative death',
          Target.INC_DEATH: 'incident death'}



def plot_results(fips, forecast_date, target, observations, pdf):
    """

    """

    df = pd.read_csv(f'{REPORT_FOLDER}/{forecast_date}_{TEAM}_{MODEL}_{fips}.csv')

    results = defaultdict(list)

    for unit in ['day', 'wk']:
        df_by_unit = df[df.target.str.contains(f'{unit} ahead {target.value}')].copy()
        for target_name in df_by_unit.target.unique():
            sub_df = df_by_unit[df_by_unit.target == target_name]
            sub_df['quantile'] = sub_df['quantile'].apply(lambda x: float(x))
            sub_df['pdf'] = np.append(sub_df['quantile'].values[0], np.diff(sub_df['quantile']))
            results[unit].append(pd.DataFrame({
                'target_end_date': [sub_df.iloc[0]['target_end_date']],
                'expected': [(sub_df['pdf'] * sub_df['value']).sum()],
                'ci_025': [float(sub_df[sub_df['quantile'] == 0.025]['value'])],
                'ci_975': [float(sub_df[sub_df['quantile'] == 0.975]['value'])]}))

    for unit in ['day', 'wk']:
        results[unit] = pd.concat(results[unit])
        results[unit]['target_end_date'] = pd.to_datetime(results[unit]['target_end_date'])

    plt.figure(figsize=(10, 6))
    for n, unit in enumerate(['day', 'wk']):
        plt.subplot(1, 2, n+1)
        plt.plot(pd.to_datetime(observations[target.value][unit].index),
                 observations[target.value][unit].values,
                 label='observed')

        plt.plot(results[unit]['target_end_date'], results[unit]['expected'], marker='o')
        plt.fill_between(x=results[unit]['target_end_date'],
                         y1=results[unit]['ci_025'],
                         y2=results[unit]['ci_975'],
                         label=f'CI_95',
                         alpha=0.3)
        plt.ylabel(f'{unit} ahead forecast')
        plt.xlabel('time')
        plt.legend()
        plt.xticks(rotation=45)


    plt.suptitle('/'.join([us.states.lookup(fips).name, fips, LABELS[target]]))
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    pdf.savefig()


def run_all():
    df_whitelist = load_data.load_whitelist()
    df_whitelist = df_whitelist[df_whitelist['inference_ok'] == True]
    fips_list = list(df_whitelist['fips'].str[:2].unique())[:5]
    output_path = f'{REPORT_FOLDER}/report_{datetime.strftime(FORECAST_DATE, DATE_FORMAT)}.pdf'
    pdf = backend_pdf.PdfPages(output_path)
    for fips in fips_list:
        observations = load_and_aggregate_observations(fips,
                                                       units=[ForecastTimeUnit.DAY, ForecastTimeUnit.WK],
                                                       targets=[Target.CUM_DEATH, Target.INC_DEATH],
                                                       smooth=False)
        for target in [Target.CUM_DEATH, Target.INC_DEATH]:
            plot_results(fips, datetime.strftime(FORECAST_DATE, DATE_FORMAT),
                         target, observations, pdf)
    pdf.close()
