import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go
import us
import numpy as np
from pyseir import load_data
from datetime import datetime, timedelta
from pyseir.inference import fit_results
from pyseir.parameters.parameter_ensemble_generator import ParameterEnsembleGenerator
from jupyterlab_dash import AppViewer
viewer = AppViewer()


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
county_metadata = load_data.load_county_metadata()
t0 = datetime(day=1, month=1, year=2020)


def get_hospitalization_data(fips):
    if len(fips) == 2:  # State FIPS are 2 digits
        hospital_times, hospitalizations, hospitalization_data_type = \
            load_data.load_hospitalization_data_by_state(us.states.lookup(fips).abbr, t0=t0)
    else:
        hospital_times, hospitalizations, hospitalization_data_type = \
            load_data.load_hospitalization_data(fips, t0=t0)

    if hospitalization_data_type:
        hospital_dates = [t0 + timedelta(days=int(t)) for t in hospital_times]

    default_parameters = ParameterEnsembleGenerator(
        fips=fips,
        N_samples=500,
        t_list=np.linspace(0, 365, 366)
    ).get_average_seir_parameters()

    if hospitalization_data_type is load_data.HospitalizationDataType.CUMULATIVE_HOSPITALIZATIONS:
        los_general = default_parameters['hospitalization_length_of_stay_general']
        los_icu = default_parameters['hospitalization_length_of_stay_icu']
        hosp_rate_general = default_parameters['hospitalization_rate_general']
        hosp_rate_icu = default_parameters['hospitalization_rate_icu']
        icu_rate = hosp_rate_icu / hosp_rate_general
        new_admissions = np.diff(hospitalizations)
        flow_out_of_hosp = ((1 - icu_rate) / los_general + icu_rate / los_icu)

        hospitalizations = [0]
        for i, new_hosp in enumerate(new_admissions):
            hospitalizations.append(hospitalizations[i-1] * (1 - flow_out_of_hosp) + new_hosp)

    return hospital_dates, hospitalizations


app.layout = html.Div([
    html.Div([
        html.H1('Covid Act Now Telemetry', style={'halign': 'center'}),
        html.Div([
            dcc.Dropdown(
                id='state',
                options=[{'label': state.name, 'value': state.fips} for state in us.STATES],
                value='06'
            ),
            dcc.RadioItems(
                id='yaxis-type',
                options=[{'label': i, 'value': i} for i in ['Linear', 'Log']],
                value='Log',
                labelStyle={'display': 'inline-block'}
            )
        ],
        style={'width': '48%', 'display': 'inline-block'}),

        html.Div([
                dcc.Dropdown(
                    id='county',
                    options=[{'label': f"{c['county']}, {c['state']} ({c['fips']})", 'value': c['fips']}
                             for idx, c in county_metadata.iterrows()],
                    value='06037'
                ),
            ],
            style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
    ]),

    html.Div(
        [
            html.Div([dcc.Graph(id='case_graphic')], style={'width': '60%', 'display': 'inline-block'}),
            html.Div([dcc.Graph(id='rt_graph')], style={'width': '38%', 'float': 'right', 'display': 'inline-block'})
         ],
    ),

    html.Div(
        [
            html.Div([dcc.Graph(id='hosp_graphic')], style={'width': '48%', 'height': '300px', 'display': 'inline-block'}),
            #html.Div([dcc.Graph(id='icu_graphic')], style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
         ],
    ),

])

# Callbacks
@app.callback(
    dash.dependencies.Output('case_graphic', 'figure'),
    [dash.dependencies.Input('state', 'value'),
     dash.dependencies.Input('yaxis-type', 'value')])
def update_case_graph(fips, yaxis_type):
    times, new_cases_observed, new_deaths_observed = load_data.load_new_case_data_by_state(fips, t0)
    dates = [(t0 + timedelta(days=t)).date().isoformat() for t in times]

    ensemble_results = load_data.load_ensemble_results(fips)
    fit_result = fit_results.load_inference_result(fips)
    model = ensemble_results['suppression_policy__inferred']
    model_dates = [t0 + timedelta(days=fit_result['t0']) + timedelta(days=t) for t in model['t_list']]

    total_infections = np.array(model['total_new_infections']['ci_50'])
    new_cases_model = np.array(model['total_new_infections']['ci_50']) * fit_result['test_fraction']
    new_deaths_model = np.array(model['total_deaths_per_day']['ci_50'])

    layout = {
        'data': [
            go.Scatter(
                x=dates,
                y=new_cases_observed,
                mode='markers',
                marker={
                    'size': 7,
                    'opacity': .4,
                    'color': 'steelblue',
                },
                name='New Cases Observed'
            ),

            go.Scatter(
                x=dates,
                y=new_deaths_observed,
                mode='markers',
                marker={
                    'size': 7,
                    'opacity': .4,
                    'color': 'firebrick',
                    'line': {'width': 0},
                    'symbol': 'diamond',
                },
                name='New Deaths Observed'
            ),

            go.Line(
                x=model_dates,
                y=total_infections,
                name='Estimated Actual Infections',
                line=dict(
                    color='steelblue',
                    width=2,
                    dash='dash'

                )
            ),

            go.Line(
                x=model_dates,
                y=new_cases_model,
                name='Estimated Tested Cases',
                line=dict(
                    color='steelblue',
                    width=3
                )
            ),

            go.Line(
                x=model_dates,
                y=new_deaths_model,
                name='Estimated Deaths',
                line=dict(
                    color='firebrick',
                    width=3,
                )
            ),
        ],

        'layout': go.Layout(
            title='New Case and Death Counts vs Inferred Model',
            xaxis=dict(
                title='',
                range=(min(dates), (datetime.today() + timedelta(days=60)).isoformat()),
            ),
            yaxis=dict(
                title='',
                type='linear' if yaxis_type == 'Linear' else 'log',
                range=(np.log10(.9), np.log10(total_infections.max())) if yaxis_type == 'Log' else (0, total_infections.max())
            ),
            margin={'l': 40, 'b': 40, 't': 40, 'r': 0},
            hovermode='closest',
            legend=dict(x=0.02, y=0.98, bgcolor='rgba(255, 255, 255, 0.5)')
        )
    }

    return layout


@app.callback(
    dash.dependencies.Output('rt_graph', 'figure'),
    [dash.dependencies.Input('state', 'value')])
def update_rt_graph(fips):

    rt_df = fit_results.load_Rt_result(fips)
    dates = [datetime.utcfromtimestamp(d) for d in fit_results.load_Rt_result('06').index.values.astype(np.int64) // 10**9]
    new_cases_rt = rt_df['Rt_MAP__new_cases']

    if 'Rt_MAP__new_deaths' in rt_df:
        new_deaths_rt = rt_df['Rt_MAP__new_deaths']
    else:
        new_deaths_rt = None

    layout = {
        'data': [
            go.Line(
                x=dates,
                y=new_cases_rt,
                name='Case-Based R(t)',
                line=dict(color='steelblue', alpha=0.5)
            ),


        ],

        'layout': go.Layout(
            title='Effective Reproduction Number R(t)',
            xaxis=dict(
                title='',
                range=(min(dates), (datetime.today() + timedelta(days=14)).isoformat()),
            ),
            yaxis=dict(
                type='linear',
                title='Effective Reproduction Number R(t)',
            ),
            margin={'l': 40, 'b': 40, 't': 40, 'r': 0},
            hovermode='x',
            legend=dict(x=0.02, y=0.02, bgcolor='rgb(255, 255, 255, 0.5)')

        )
    }

    if new_deaths_rt is not None:
        layout['data'] += [
            go.Line(
                x=dates,
                y=new_deaths_rt,
                text='',
                name='Deaths-Based R(t)',
                line=dict(color='firebrick')
            ),

            go.Line(
                x=dates,
                y=np.nanmean([new_cases_rt, new_deaths_rt], axis=0),
                text='Mean of Case and Deaths',
                name='Composite',
                line=dict(color='black', width=4)
            )
        ]
    return layout



@app.callback(
    dash.dependencies.Output('hosp_graphic', 'figure'),
    [dash.dependencies.Input('state', 'value'),
     dash.dependencies.Input('yaxis-type', 'value')])
def update_hosp_graph(fips, yaxis_type):

    dates, new_hospitalizations = get_hospitalization_data(fips)

    ensemble_results = load_data.load_ensemble_results(fips)
    fit_result = fit_results.load_inference_result(fips)
    model = ensemble_results['suppression_policy__inferred']
    model_dates = [t0 + timedelta(days=fit_result['t0']) + timedelta(days=t) for t in model['t_list']]

    actual_hosp = np.array(model['HGen']['ci_50'])
    observed_hosp = np.array(model['HGen']['ci_50']) * fit_result['hosp_fraction']

    actual_hosp_icu = np.array(model['HICU']['ci_50'])
    observed_hosp_icu = np.array(model['HICU']['ci_50']) * fit_result['hosp_fraction']

    layout = {
        'data': [
            go.Scatter(
                x=dates,
                y=new_hospitalizations,
                mode='markers',
                marker={
                    'size': 7,
                    'opacity': .4,
                    'color': 'goldenrod',
                    'line': {'width': 0},
                    'symbol': 'diamond',
                },
                name='Observed Hospitalizations'
            ),

            go.Line(
                x=model_dates,
                y=observed_hosp,
                name='Estimated Reported Hospital Occupancy',
                text='Many states are under reporting hospitalizations. Our model tries to account for this by fitting the reporting rate.',
                line=dict(
                    color='darkseagreen',
                    width=2,
                )
            ),


            # go.Line(
            #     x=model_dates,
            #     y=actual_hosp_icu,
            #     name='Estimated Total ICU Occupancy',
            #     line=dict(
            #         color='goldenrod',
            #         width=4,
            #         dash='dash'
            #     )
            # ),

            go.Line(
                x=model_dates,
                y=observed_hosp_icu,
                name='Estimated Reported ICU Occupancy',
                line=dict(
                    color='goldenrod',
                    width=4,
                    dash='dash'
                )
            ),

        ],

        'layout': go.Layout(
            title='Hospitalization Capacity',
            xaxis=dict(
                title='',
                range=(min(model_dates), (datetime.today() + timedelta(days=60)).isoformat()),
            ),
            yaxis=dict(
                title='',
                type='linear' if yaxis_type == 'Linear' else 'log',
            ),
            margin={'l': 40, 'b': 40, 't': 40, 'r': 0},
            hovermode='closest',
            legend=dict(x=.5, y=0.0, bgcolor='rgba(255, 255, 255, 0.5)')
        )
    }

    return layout


if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0')
