from pyseir.fit_results import load_mle_model, load_inference_result
from pyseir import load_data
import pandas as pd
from pyseir.utils import REF_DATE
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import us
import matplotlib
from pyseir.cdc.utils import Target

def plot_results(fips, forecast_date, target, pdf):
    times, _, death = load_data.load_new_case_data_by_state(fips, t0=REF_DATE)
    df = pd.read_csv(f'output/pyseir/cdc/{forecast_date}_CovidActNow_SEIR_CAN_{fips}.csv')
    df = df[df.target.str.contains(f'day ahead {target.value}')]

    results = list()

    for target_name in df.target.unique():
        sub_df = df[df.target == target_name]
        sub_df['quantile'] = sub_df['quantile'].apply(lambda x: float(x))
        sub_df['pdf'] = np.append(sub_df['quantile'].values[0], np.diff(sub_df['quantile']))
        results.append(pd.DataFrame({
            'target_end_date': [sub_df.iloc[0]['target_end_date']],
            'expected': [(sub_df['pdf'] * sub_df['value']).sum()],
            'ci_025': [float(sub_df[sub_df['quantile'] == 0.025]['value'])],
            'ci_975': [float(sub_df[sub_df['quantile'] == 0.975]['value'])]}))

    results = pd.concat(results)
    results['target_end_date'] = pd.to_datetime(results['target_end_date'])

    plt.figure()
    if target is Target.CUM_DEATH:
        plt.plot([REF_DATE + timedelta(t) for t in times], death.cumsum(), label='observed')
    plt.plot(results['target_end_date'], results['mean'])
    plt.fill_between(x=results['target_end_date'],
                     y1=results['ci_025'], y2=results['ci_975'],
                     label='CI_95', alpha=0.3)
    plt.legend()

    plt.title('\n'.join([us.states.lookup(fips).name, fips, 'cumulative death']))

    plt.tight_layout()
    pdf.savefig()

def run_all():
    df_whitelist = load_data.load_whitelist()
    df_whitelist = df_whitelist[df_whitelist['inference_ok'] == True]
    fips_list = list(df_whitelist['fips'].str[:2].unique())

    om = OutputMapper(fips='06')
    for target in
    output_path = 'output/pyseir/cdc/report_cum_death.pdf'
    pdf = matplotlib.backends.backend_pdf.PdfPages(output_path)
    for fips in fips_list:
        plot_results(fips, forecast_date, target, pdf)
    pdf.close()
