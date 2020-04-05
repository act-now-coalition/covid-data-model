import sys
import click
import us
import logging
from pyseir.load_data import cache_all_data
from pyseir.inference.initial_conditions_fitter import generate_start_times_for_state
from pyseir.ensembles.ensemble_runner import run_state
from pyseir.reports.state_report import StateReport
from pyseir.inference import model_fitter_mle


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
        for state in us.states.STATES:
            _impute_start_dates(state)


def _run_mle_fits(state=None):
    if state:
        model_fitter_mle.run_state(state.title())
    else:
        for state in us.states.STATES:
            run_mle_fits(state)


def _run_ensembles(state=None, ensemble_kwargs=dict()):
    if state:
        run_state(state, ensemble_kwargs=ensemble_kwargs)
    else:
        for state in us.states.STATES:
            run_state(state, ensemble_kwargs=ensemble_kwargs)


def _generate_state_reports(state=None):
    if state:
        report = StateReport(state.title())
        report.generate_report()
    else:
        for state in us.states.STATES:
            _generate_state_reports(state.title())


def _run_all(state=None):
    exceptions = []
    cache_all_data()

    if state:
        _impute_start_dates(state.title())
        _run_mle_fits(state)
        _run_ensembles(state.title())
        _generate_state_reports(state.title())
    else:
        for state in us.states.STATES:
            try:
                _generate_state_reports(state.name)
            except ValueError as e:
                exceptions.append(exceptions)
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
@click.option('--run-mode', default='default', help='State to generate files for. If no state is given, all states are computed.')
def run_ensembles(state, run_mode):
    _run_ensembles(state, run_mode)


@entry_point.command()
@click.option('--state', default='', help='State to generate files for. If no state is given, all states are computed.')
def generate_state_report(state):
    _generate_state_reports(state)


@entry_point.command()
@click.option('--state', default='', help='State to generate files for. If no state is given, all states are computed.')
def run_all(state=None):
    _run_all(state)
