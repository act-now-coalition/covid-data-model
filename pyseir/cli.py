from typing import Dict, List
import sys
import os
import click
import us
import logging
import pandas as pd
from covidactnow.datapublic import common_init

from structlog_sentry import SentryProcessor
from multiprocessing import Pool
from functools import partial
from libs.datasets import dataset_cache
from pyseir.inference.initial_conditions_fitter import generate_start_times_for_state
from pyseir.inference import infer_rt as infer_rt_module
from pyseir.ensembles import ensemble_runner
from pyseir.reports.state_report import StateReport
from pyseir.inference import model_fitter
from pyseir.deployment.webui_data_adaptor_v1 import WebUIDataAdaptorV1
from libs.datasets import combined_datasets
from libs.us_state_abbrev import ABBREV_US_STATE
from pyseir.inference.whitelist_generator import WhitelistGenerator
from pyseir.inference import forecast_rt

from libs.datasets.timeseries import TimeseriesDataset
from libs.datasets.dataset_utils import AggregationLevel


sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), ".."))

root = logging.getLogger()

DEFAULT_RUN_MODE = "can-inference-derived"
ALL_STATES = [getattr(state_obj, "name") for state_obj in us.STATES]


def _cache_global_datasets():
    # Populate cache for combined latest and timeseries.  Caching pre-fork
    # will make sure cache is populated for subprocesses.  Return value
    # is not needed as the only goal is to populate the cache.
    combined_datasets.build_us_latest_with_all_fields()
    combined_datasets.build_us_timeseries_with_all_fields()


@click.group()
def entry_point():
    """Basic entrypoint for cortex subcommands"""
    dataset_cache.set_pickle_cache_dir()
    common_init.configure_logging()


def _generate_whitelist():
    gen = WhitelistGenerator()
    gen.generate_whitelist()


def _impute_start_dates(state=None, states_only=False):
    if states_only:
        raise NotImplementedError(
            "Impute start dates does not yet implement support for states_only."
        )

    if state:
        generate_start_times_for_state(state=state)
    else:
        for state_name in ALL_STATES:
            _impute_start_dates(state_name)


def _infer_rt(state=None, states_only=False):
    if state:
        infer_rt_module.run_state(state=state, states_only=states_only)
    else:
        for state_name in ALL_STATES:
            _infer_rt(state=state_name, states_only=states_only)


def _run_mle_fits(state=None, states_only=False):
    _cache_global_datasets()
    if state:
        model_fitter.run_state(state, states_only=states_only)
    else:
        for state_name in ALL_STATES:
            _run_mle_fits(state=state_name, states_only=states_only)


def _run_ensembles(state=None, ensemble_kwargs=dict(), states_only=False):
    if state:
        ensemble_runner.run_state(state, ensemble_kwargs=ensemble_kwargs, states_only=states_only)
    else:
        for state_name in ALL_STATES:
            ensemble_runner.run_state(
                state_name, ensemble_kwargs=ensemble_kwargs, states_only=states_only
            )


def _generate_state_reports(state=None):
    if state:
        report = StateReport(state)
        report.generate_report()
    else:
        for state_name in ALL_STATES:
            _generate_state_reports(state_name)


def _map_outputs(
    state=None, output_interval_days=1, states_only=False, output_dir=None, run_mode="default",
):
    output_interval_days = int(output_interval_days)
    _cache_global_datasets()
    if state:
        web_ui_mapper = WebUIDataAdaptorV1(
            state,
            output_interval_days=output_interval_days,
            run_mode=run_mode,
            output_dir=output_dir,
        )
        web_ui_mapper.generate_state(
            whitelisted_county_fips=[], states_only=states_only,
        )
    else:
        for state_name in ALL_STATES:
            _map_outputs(
                state_name,
                output_interval_days,
                states_only=states_only,
                run_mode=run_mode,
                output_dir=output_dir,
            )


def _state_only_pipeline(
    state,
    run_mode=DEFAULT_RUN_MODE,
    generate_reports=False,
    output_interval_days=1,
    output_dir=None,
):
    states_only = True
    _infer_rt(state, states_only=states_only)
    # _run_mle_fits(state, states_only=states_only)
    # _run_ensembles(
    #    state,
    #    ensemble_kwargs=dict(run_mode=run_mode, generate_report=generate_reports),
    #    states_only=states_only,
    # )
    if generate_reports:
        _generate_state_reports(state)
    # remove outputs atm. just output at the end
    _map_outputs(
        state,
        output_interval_days,
        states_only=states_only,
        output_dir=output_dir,
        run_mode=run_mode,
    )


def build_counties_to_run_per_state(states: List[str], fips: str = None) -> Dict[str, str]:
    """Builds mapping from fips to state of counties to run.

    Restricts counties to those in the county whitelist.

    Args:
        states: List of states to run on.
        fips: Optional county fips code to restrict results to.

    Returns: Map of counties to run with associated state.
    """
    # Build List of Counties
    all_county_fips = {}
    for state in states:
        state_county_fips = model_fitter.build_county_list(state)
        county_fips_per_state = {fips: state for fips in state_county_fips}

        if not fips:
            all_county_fips.update(county_fips_per_state)
            continue

        if fips in county_fips_per_state:
            root.info(f"Found {fips}, restricting run to found fips")
            all_county_fips.update({fips: state})

    return all_county_fips


def _build_all_for_states(
    states=[],
    run_mode=DEFAULT_RUN_MODE,
    generate_reports=False,
    output_interval_days=4,
    skip_download=False,
    output_dir=None,
    skip_whitelist=False,
    states_only=False,
    fips=None,
):
    print("running forecast")
    forecast_rt.external_run_forecast()
    print("ran forecast")
    return
    # prepare data
    _cache_global_datasets()
    print("saving data")
    all_data = combined_datasets.build_us_timeseries_with_all_fields().get_data(
        AggregationLevel.STATE, country="USA"
    )
    print("done")
    print(all_data)
    print("making csv")
    all_data.to_csv("pyseir_combined_datasets_USA.csv")
    print("made csv")

    if not skip_whitelist:
        _generate_whitelist()

    # do everything for just states in parallel
    with Pool(maxtasksperchild=1) as p:
        states_only_func = partial(
            _state_only_pipeline,
            run_mode=run_mode,
            generate_reports=generate_reports,
            output_interval_days=output_interval_days,
            output_dir=output_dir,
        )
        p.map(states_only_func, states)

    if states_only:
        root.info("Only executing for states. returning.")
        return

    all_county_fips = build_counties_to_run_per_state(states, fips=fips)

    with Pool(maxtasksperchild=1) as p:
        # calculate calculate county inference
        p.map(infer_rt_module.run_county, all_county_fips.keys())
        # calculate model fit
        root.info(f"executing model for {len(all_county_fips)} counties")
        fitters = p.map(model_fitter.execute_model_for_fips, all_county_fips.keys())

        df = pd.DataFrame([fit.fit_results for fit in fitters if fit])
        df["state"] = df.fips.replace(all_county_fips)
        df["mle_model"] = [fit.mle_model for fit in fitters if fit]
        df.index = df.fips

        state_dfs = [state_df for name, state_df in df.groupby("state")]
        p.map(model_fitter._persist_results_per_state, state_dfs)

        # calculate ensemble
        root.info(f"running ensemble for {len(all_county_fips)} counties")
        ensemble_func = partial(
            ensemble_runner._run_county,
            ensemble_kwargs=dict(run_mode=run_mode, generate_report=generate_reports),
        )
        p.map(ensemble_func, all_county_fips.keys())

    # output it all
    output_interval_days = int(output_interval_days)
    _cache_global_datasets()

    root.info(f"outputting web results for states and {len(all_county_fips)} counties")
    # does not parallelize well, because web_ui mapper doesn't serialize efficiently
    # TODO: Remove intermediate artifacts and paralellize artifacts creation better
    # Approximately 40% of the processing time is taken on this step
    for state in states:
        web_ui_mapper = WebUIDataAdaptorV1(
            state,
            output_interval_days=output_interval_days,
            run_mode=run_mode,
            output_dir=output_dir,
        )
        web_ui_mapper.generate_state(
            whitelisted_county_fips=[k for k, v in all_county_fips.items() if v == state],
            states_only=False,
        )

    return


@entry_point.command()
@click.option(
    "--state",
    default="",
    help="State to generate files for. If no state is given, all states are computed.",
)
@click.option("--states-only", default=False, is_flag=True, type=bool, help="Only model states")
def impute_start_dates(state, states_only):
    _impute_start_dates(state, states_only)


@entry_point.command()
def generate_whitelist():
    _generate_whitelist()


@entry_point.command()
@click.option(
    "--state",
    default="",
    help="State to generate files for. If no state is given, all states are computed.",
)
@click.option("--states-only", default=False, is_flag=True, type=bool, help="Only model states")
def infer_rt(state, states_only):
    _infer_rt(state, states_only=states_only)


@entry_point.command()
@click.option(
    "--state",
    default="",
    help="State to generate files for. If no state is given, all states are computed.",
)
@click.option("--states-only", default=False, is_flag=True, type=bool, help="Only model states")
def run_mle_fits(state, states_only):
    _run_mle_fits(state, states_only=states_only)


@entry_point.command()
@click.option(
    "--state",
    default="",
    help="State to generate files for. If no state is given, all states are computed.",
)
@click.option(
    "--generate-reports",
    default=False,
    is_flag=True,
    type=bool,
    help="If False, skip pdf report generation.",
)
@click.option(
    "--run-mode",
    default=DEFAULT_RUN_MODE,
    type=click.Choice([run_mode.value for run_mode in ensemble_runner.RunMode]),
    help="State to generate files for. If no state is given, all states are computed.",
)
@click.option("--states-only", default=False, is_flag=True, type=bool, help="Only model states")
def run_ensembles(state, run_mode, generate_reports, states_only):
    _run_ensembles(
        state,
        ensemble_kwargs=dict(run_mode=run_mode, generate_report=generate_reports),
        states_only=states_only,
    )


@entry_point.command()
@click.option(
    "--state",
    default="",
    help="State to generate files for. If no state is given, all states are computed.",
)
def generate_state_report(state):
    _generate_state_reports(state)


@entry_point.command()
@click.option(
    "--state",
    default="",
    help="State to generate files for. If no state is given, all states are computed.",
)
@click.option(
    "--output-interval-days",
    default=1,
    type=int,
    help="Number of days between outputs for the WebUI payload.",
)
@click.option(
    "--run-mode",
    default=DEFAULT_RUN_MODE,
    type=click.Choice([run_mode.value for run_mode in ensemble_runner.RunMode]),
    help="State to generate files for. If no state is given, all states are computed.",
)
@click.option("--states-only", default=False, is_flag=True, type=bool, help="Only model states")
def map_outputs(state, output_interval_days, run_mode, states_only):
    _map_outputs(
        state,
        output_interval_days=int(output_interval_days),
        run_mode=run_mode,
        states_only=states_only,
    )


@entry_point.command()
@click.option(
    "--states",
    "-s",
    multiple=True,
    help="a list of states to generate files for. If no state is given, all states are computed.",
)
@click.option(
    "--run-mode",
    default=DEFAULT_RUN_MODE,
    type=click.Choice([run_mode.value for run_mode in ensemble_runner.RunMode]),
    help="State to generate files for. If no state is given, all states are computed.",
)
@click.option(
    "--generate-reports",
    default=False,
    type=bool,
    is_flag=True,
    help="If False, skip pdf report generation.",
)
@click.option(
    "--output-interval-days",
    default=1,
    type=int,
    help="Number of days between outputs for the WebUI payload.",
)
@click.option(
    "--skip-whitelist", default=False, is_flag=True, type=bool, help="Skip the whitelist phase.",
)
@click.option(
    "--fips",
    help=(
        "County level fips code to restrict runs to. "
        "This does not restrict the states that run, so also specifying states with "
        "`--states` is recommended."
    ),
)
@click.option("--states-only", is_flag=True, help="If set, only runs on states.")
@click.option("--output-dir", default=None, type=str, help="Directory to deploy webui output.")
def build_all(
    states,
    run_mode,
    generate_reports,
    output_interval_days,
    output_dir,
    skip_whitelist,
    states_only,
    fips,
):
    # split columns by ',' and remove whitespace
    states = [c.strip() for c in states]
    # Convert abbreviated states to the full state name, allowing states passed in
    # to be the full name or the abbreviated name.
    states = [ABBREV_US_STATE.get(state, state) for state in states]
    states = [state for state in states if state in ALL_STATES]
    if not len(states):
        states = ALL_STATES

    _build_all_for_states(
        states=states,
        run_mode=DEFAULT_RUN_MODE,
        generate_reports=generate_reports,
        output_interval_days=output_interval_days,
        output_dir=output_dir,
        skip_whitelist=skip_whitelist,
        states_only=states_only,
        fips=fips,
    )


if __name__ == "__main__":
    try:
        entry_point()  # pylint: disable=no-value-for-parameter
    except Exception:
        # According to https://github.com/getsentry/sentry-python/issues/480 Sentry is expected
        # to create an event when this is called.
        logging.exception("Exception reached __main__")
        raise
