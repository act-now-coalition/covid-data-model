import sys, os
import click
import us
import logging
from pyseir.load_data import cache_all_data
from pyseir.inference.initial_conditions_fitter import generate_start_times_for_state
from pyseir.ensembles.ensemble_runner import run_state
from pyseir.reports.state_report import StateReport
from pyseir.inference import model_fitter_mle
from pyseir.deployment.webui_data_adaptor_v1 import WebUIDataAdaptorV1
sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))

root = logging.getLogger()
root.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(filename)s - %(lineno)d - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
root.addHandler(handler)


@click.group()
def entry_point():
    """Basic entrypoint for cortex subcommands"""
    pass


@entry_point.command()
def download_data():
    cache_all_data()


def _impute_start_dates(state=None):
    if state:
        generate_start_times_for_state(state=state.title())
    else:
        for state_obj in us.states.STATES:
            _impute_start_dates(state_obj.name)


def _run_mle_fits(state=None):
    if state:
        model_fitter_mle.run_state(state.title())
    else:
        for state_obj in us.states.STATES:
            run_mle_fits(state_obj.name)


def _run_ensembles(state=None, ensemble_kwargs=dict()):
    if state:
        run_state(state, ensemble_kwargs=ensemble_kwargs)
    else:
        for state_obj in us.states.STATES:
            run_state(state_obj.name, ensemble_kwargs=ensemble_kwargs)


def _generate_state_reports(state=None):
    if state:
        report = StateReport(state.title())
        report.generate_report()
    else:
        for state_obj in us.states.STATES:
            _generate_state_reports(state_obj.name)


def _map_outputs(state=None, output_interval_days=4):
    output_interval_days = int(output_interval_days)
    if state:
        web_ui_mapper = WebUIDataAdaptorV1(state, output_interval_days=output_interval_days)
        web_ui_mapper.generate_state()
    else:
        for state_obj in us.states.STATES:
            _map_outputs(state_obj.name, output_interval_days)


def _run_all(state=None, run_mode='default', generate_reports=True, output_interval_days=4, skip_download=False):
    exceptions = []

    if not skip_download:
        cache_all_data()

    if state:
        _impute_start_dates(state.title())
        _run_mle_fits(state)
        _run_ensembles(state.title(),
                       ensemble_kwargs=dict(
                           run_mode=run_mode,
                           generate_report=generate_reports
                       ))
        if generate_reports:
            _generate_state_reports(state.title())
        _map_outputs(state, output_interval_days)
    else:
        for state_obj in us.states.STATES:
            try:
                _run_all(state_obj.name, run_mode, generate_reports, output_interval_days, skip_download=True)
            except ValueError as e:
                exceptions.append(e)
    for exception in exceptions:
        logging.critical(exception)


@entry_point.command()
@click.option('--state', default='', help='State to generate files for. If no state is given, all states are computed.')
def impute_start_dates(state):
    _impute_start_dates(state)


@entry_point.command()
@click.option('--state', default='', help='State to generate files for. If no state is given, all states are computed.')
def run_mle_fits(state):
    _run_mle_fits(state)


@entry_point.command()
@click.option('--state', default='', help='State to generate files for. If no state is given, all states are computed.')
@click.option('--generate-reports', default=True, type=bool, help='If False, skip pdf report generation.')
@click.option('--run-mode', default='default', help='State to generate files for. If no state is given, all states are computed.')
def run_ensembles(state, run_mode, generate_reports):
    _run_ensembles(state, ensemble_kwargs=dict(run_mode=run_mode, generate_report=generate_reports))


@entry_point.command()
@click.option('--state', default='', help='State to generate files for. If no state is given, all states are computed.')
def generate_state_report(state):
    _generate_state_reports(state)


@entry_point.command()
@click.option('--state', default='', help='State to generate files for. If no state is given, all states are computed.')
@click.option('--output-interval-days', default='', type=int, help='Number of days between outputs for the WebUI payload.')
def map_outputs(state, output_interval_days):
    _map_outputs(state, output_interval_days=int(output_interval_days))


@entry_point.command()
@click.option('--state', default=None, help='State to generate files for. If no state is given, all states are computed.')
@click.option('--run-mode', default='default',type=str, help='State to generate files for. If no state is given, all states are computed.')
@click.option('--generate-reports', default=True, type=bool, help='If False, skip pdf report generation.')
@click.option('--output-interval-days', default='4', type=int, help='Number of days between outputs for the WebUI payload.')
def run_all(state, run_mode, generate_reports, output_interval_days):
    _run_all(state, run_mode, generate_reports, output_interval_days)
