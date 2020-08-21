import pathlib

import pytest
import pandas as pd
import numpy as np
import math
from random import choices, randrange
import structlog
from matplotlib import pyplot as plt
from datetime import datetime, timedelta

from pyseir.models.seir_model import (
    SEIRModel,
    steady_state_ratios,
)
from pyseir.rt.constants import InferRtConstants

# rom pyseir.utils import get_run_artifact_path, RunArtifact
from test.mocks.inference import load_data
from test.mocks.inference.load_data import RateChange

TEST_OUTPUT_DIR = pathlib.Path(__file__).parent.parent / "output" / "test_results"


def make_tlist(num_days):
    return np.linspace(0, num_days, num_days + 1)


def create_standard_model(r0, sup, days=100, ratios=None):
    """
    Creates a standard model for testing purposes. Supports 
    - customization of r0 and suppresssion policy
    - optionally supply ratios for (non infection) compartments that matter
    """
    hosp_rate_general = 0.025
    initial_infected = 1000
    N = 10000000

    if ratios is None:
        model = SEIRModel(
            N=N,
            t_list=make_tlist(days),
            suppression_policy=sup,
            R0=r0,
            # gamma=0.7,
            # hospitalization_rate_general=hosp_rate_general,
            # hospitalization_rate_icu=0.3 * hosp_rate_general,
            A_initial=0.7 * initial_infected,
            I_initial=initial_infected,
            beds_general=N / 1000,
            beds_ICU=N / 1000,
            ventilators=N / 1000,
        )
    else:
        (E, I, A, HGen, HICU, HVent, D) = ratios
        model = SEIRModel(
            N=N,
            t_list=make_tlist(days),
            suppression_policy=sup,
            R0=r0,
            E_initial=E * initial_infected,
            A_initial=A * initial_infected,
            I_initial=initial_infected,
            HGen_initial=HGen * initial_infected,
            HICU_initial=HICU * initial_infected,
            HICUVent_initial=HVent * initial_infected,
            beds_general=N / 1000,
            beds_ICU=N / 1000,
            ventilators=N / 1000,
        )
    return model


def test_run_model_orig():
    def sup(t):
        return 1.0 if t < 50 else 0.6

    model = create_standard_model(1.4, sup, 200)

    model.run()

    fig = model.plot_results(alternate_plots=True)
    fig.savefig(TEST_OUTPUT_DIR / "test_run_model_orig.pdf", bbox_inches="tight")


def test_restart_existing_model_from_ratios():
    ratios = steady_state_ratios(1.4)

    def sup(t):
        return 1.0

    model = create_standard_model(1.4, sup, 50, ratios)
    model.run()

    fig = model.plot_results(alternate_plots=True)
    fig.savefig(TEST_OUTPUT_DIR / "test_restart_from_ratios.pdf", bbox_inches="tight")
