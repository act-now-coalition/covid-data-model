import itertools
import sys
import os
from typing import Optional
from typing import List
import logging
from multiprocessing import Pool
from functools import partial

import us
import pandas as pd
import click
from covidactnow.datapublic.common_fields import CommonFields

from covidactnow.datapublic import common_init
from libs import pipeline
from pyseir.deployment import webui_data_adaptor_v1
from pyseir.inference import whitelist
from pyseir.rt import infer_rt
from pyseir.ensembles import ensemble_runner
from pyseir.inference import model_fitter
from pyseir.deployment.webui_data_adaptor_v1 import WebUIDataAdaptorV1
from libs.datasets import combined_datasets
import pyseir.utils
from pyseir.inference.whitelist import WhitelistGenerator
from pyseir.rt.utils import NEW_ORLEANS_FIPS

sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), ".."))

root = logging.getLogger()

ALL_STATES: List[str] = [state_obj.abbr for state_obj in us.STATES] + ["PR"]


def _cache_global_datasets():
    # Populate cache for combined latest and timeseries.  Caching pre-fork
    # will make sure cache is populated for subprocesses.  Return value
    # is not needed as the only goal is to populate the cache.
    combined_datasets.load_us_latest_dataset()
    combined_datasets.load_us_timeseries_dataset()


@click.group()
def entry_point():
    """Basic entrypoint for cortex subcommands"""
    common_init.configure_logging()


def _generate_whitelist() -> pd.DataFrame:
    gen = WhitelistGenerator()
    all_us_timeseries = combined_datasets.load_us_timeseries_dataset()
    return gen.generate_whitelist(all_us_timeseries)


def _generate_infection_rate_metric(regions: List[infer_rt.RegionalInput]) -> pd.DataFrame:
    """
    Apply infer_rt.run_rt_for_fips for each region in regions and return a combined results table
    with a unique row for each region+date combination.
    """
    if not regions:
        return pd.DataFrame()

    with Pool(maxtasksperchild=10) as p:
        results = pd.concat(p.map(infer_rt.run_rt, regions))

    if results.empty:
        return pd.DataFrame()

    need_patch = set(results.fips) & set(NEW_ORLEANS_FIPS)
    if need_patch:
        logging.info("Applying New Orleans Patch")
        unpatched = results.loc[~results.fips.isin(need_patch), :]

        # Aggregate the results created so far into one timeseries of metrics in a DataFrame
        nola_rt_inferences = pyseir.rt.patches.patch_aggregate_rt_results(need_patch)

        patched: List[pd.DataFrame] = []
        for fips in need_patch:
            fips_rt_inferences = nola_rt_inferences.copy()
            fips_rt_inferences.insert(0, CommonFields.FIPS, fips)
            patched.append(fips_rt_inferences)
            # TODO(tom): Delete when no longer read
            output_path = pipeline.Region.from_fips(fips).run_artifact_path_to_write(
                pyseir.utils.RunArtifact.RT_INFERENCE_RESULT
            )
            fips_rt_inferences.to_json(output_path)
        results = pd.concat(patched + [unpatched])

    return results


def _states_region_list(state: Optional[str], default: List[str]) -> List[pipeline.Region]:
    """Create a list of Region objects containing just state or default."""
    if state:
        return [pipeline.Region.from_state(state)]
    else:
        return [pipeline.Region.from_state(s) for s in default]


def _state_only_pipeline(
    region: pipeline.Region, output_interval_days=1, output_dir=None,
) -> model_fitter.ModelFitter:
    assert region.is_state()

    infer_rt.run_rt(infer_rt.RegionalInput.from_region(region))
    fitter = model_fitter.run_state(region)
    ensembles_input = ensemble_runner.RegionalInput.from_model_fitter(fitter)
    ensemble_runner.run_region(ensembles_input)
    return fitter


def _build_all_for_states(
    states: List[str],
    output_interval_days=4,
    output_dir=None,
    skip_whitelist=False,
    states_only=False,
    fips: str = None,
):
    # prepare data
    _cache_global_datasets()

    # do everything for just states in parallel
    with Pool(maxtasksperchild=1) as p:
        states_only_func = partial(
            _state_only_pipeline, output_interval_days=output_interval_days, output_dir=output_dir,
        )
        states_regions = [pipeline.Region.from_state(s) for s in states]
        state_fitters: List[model_fitter.ModelFitter] = p.map(states_only_func, states_regions)
        state_fitter_map = {f.region: f for f in state_fitters}

    if states_only:
        root.info("Only executing for states. returning.")
        return

    # Calculate the whitelist for the infection rate metric which makes no promises
    # about it's relationship to the SEIR subset

    infection_rate_fips_whitelist = [
        infer_rt.RegionalInput.from_fips(x)
        for x in combined_datasets.load_us_latest_dataset().all_fips
        if not ("25" in x[:2] and len(x) == 5)  # Masking MA Counties (2020-08-27) due to NaNs
        and "999" not in x[-3:]  # Remove placeholder fips that have no data
    ]

    if fips:  # A single Fips string was passed as a flag. Just run for that fips.
        infer_rt_candidates = [infer_rt.RegionalInput.from_fips(fips)]
    else:  # Default to the full infection rate whitelist
        infer_rt_candidates = infection_rate_fips_whitelist

    # Separating the Infection Rate Calculations
    infection_rate_metric_df = _generate_infection_rate_metric(regions=infer_rt_candidates)
    # Note: This call currently still both writes (1) region by region JSON and (2) returns this
    # combined dataset. Once the api generation is changed to point to the combined dataset, the
    # individual objects can stop being persisted to disk. It also runs for all fips available, so
    # currently it double calculates the state fips (since the state only is called earlier in this
    # function. I would prefer to run the state only requirement instead of changing this behaviour,
    # so that the saved csv has both state and county data.

    infection_rate_metric_df.to_csv(
        path_or_buf=pyseir.utils.get_summary_artifact_path(
            pyseir.utils.SummaryArtifact.RT_METRIC_COMBINED
        ),
        index=False,
    )

    # Now calculate the pyseir dependent whitelist
    whitelist_df = _generate_whitelist()
    all_county_regions = whitelist.regions_in_states(
        states_regions, fips=fips, whitelist_df=whitelist_df
    )

    with Pool(maxtasksperchild=1) as p:
        # calculate model fit
        root.info(f"executing model for {len(all_county_regions)} counties")
        fitter_inputs = [
            model_fitter.RegionalInput.from_region(
                r, state_fitter=state_fitter_map.get(r.get_state_region())
            )
            for r in all_county_regions
        ]
        county_fitters = p.map(model_fitter.ModelFitter.run_for_region, fitter_inputs)

        # calculate ensemble
        root.info(f"running ensemble for {len(all_county_regions)} counties")
        ensemble_inputs = [
            ensemble_runner.RegionalInput.from_model_fitter(f) for f in county_fitters
        ]
        p.map(ensemble_runner.run_region, ensemble_inputs)

    # output it all
    output_interval_days = int(output_interval_days)
    _cache_global_datasets()

    root.info(f"outputting web results for states and {len(all_county_regions)} counties")

    # does not parallelize well, because web_ui mapper doesn't serialize efficiently
    # TODO: Remove intermediate artifacts and paralellize artifacts creation better
    # Approximately 40% of the processing time is taken on this step
    web_ui_mapper = WebUIDataAdaptorV1(
        output_interval_days=output_interval_days, output_dir=output_dir,
    )

    webui_inputs = [
        webui_data_adaptor_v1.RegionalInput.from_model_fitter(fitter)
        for fitter in itertools.chain(state_fitters, county_fitters)
    ]
    with Pool(maxtasksperchild=1) as p:
        p.map(web_ui_mapper.write_region, webui_inputs)


@entry_point.command()
def generate_whitelist():
    _generate_whitelist()


@entry_point.command()
@click.option(
    "--state", help="State to generate files for. If no state is given, all states are computed."
)
@click.option(
    "--states-only",
    default=False,
    is_flag=True,
    type=bool,
    help="Warning: This flag is unused and the function always defaults to only state "
    "level regions",
)
def run_infer_rt(state, states_only):
    for state in _states_region_list(state=state, default=ALL_STATES):
        infer_rt.run_rt(infer_rt.RegionalInput.from_region(state))


@entry_point.command()
@click.option(
    "--state", help="State to generate files for. If no state is given, all states are computed."
)
@click.option("--states-only", default=False, is_flag=True, type=bool, help="Only model states")
def run_mle_fits(state, states_only):
    states_regions = _states_region_list(state=state, default=ALL_STATES)

    for region in states_regions:
        model_fitter.run_state(region)

    if not states_only:
        whitelist_df = _generate_whitelist()
        for region in states_regions:
            county_regions = whitelist.regions_in_states([region], whitelist_df=whitelist_df)
            model_fitter.run_counties_of_state(county_regions)


@entry_point.command()
@click.option(
    "--states",
    "-s",
    multiple=True,
    help="a list of states to generate files for. If no state is given, all states are computed.",
)
@click.option(
    "--output-interval-days",
    default=1,
    type=int,
    help="Number of days between outputs for the WebUI payload.",
)
@click.option(
    "--skip-whitelist", default=False, is_flag=True, type=bool, help="Skip the whitelist phase."
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
    states, output_interval_days, output_dir, skip_whitelist, states_only, fips,
):
    # split columns by ',' and remove whitespace
    states = [c.strip() for c in states]
    states = [us.states.lookup(state).abbr for state in states]
    states = [state for state in states if state in ALL_STATES]
    if not len(states):
        states = ALL_STATES

    _build_all_for_states(
        states,
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
