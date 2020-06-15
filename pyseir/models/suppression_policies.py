from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from pyseir import load_data
from pyseir.inference.infer_t0 import infer_t0
from pyseir.inference import fit_results


# Fig 4 of Imperial college.
# https://www.imperial.ac.uk/media/imperial-college/medicine/sph/ide/gida-fellowships/Imperial-College-COVID19-Europe-estimates-and-NPI-impact-30-03-2020.pdf
# These are intended to act indendently, as shown by a multivariate fit from Imp. College.  The exception is lockdown which supersedes everything.
distancing_measure_suppression = {
    "stay_at_home": 0.48,
    "50_gatherings": 0.05,
    "500_gatherings": 0.02,  # Made this one up since not examined. Assume it isn't very effective at county level, esp. relative to 50 gatherings
    "self_isolation": 0.05,  # This one is not included in the policies dataset , but is in the imperial college paper. Keep it around for now..
    "public_schools": 0.18,  # Total social distancing was about
    "entertainment_gym": 0.02,
    "restaurant_dine-in": 0.03,
    "federal_guidelines": 0.03,  # Making this up as well. Probably not very effective relative to stay at home...
}


def estimate_future_suppression_from_fits(inferred_params, scenario):
    """
    Returns the future suppression level for a given R0, R_eff2 and a "future scenario". The
    "no_intervention" scenario is capped at an effective R rate of 2.5 as suggested by the CDC
    planning scenarios (https://www.cdc.gov/coronavirus/2019-ncov/hcp/planning-scenarios.html).

    Parameters
    ----------
    inferred_params: dict
        Fit results including R0 and eps2 as keys
    scenario: str
        'no_intervention', 'flatten_the_curve', 'social_distancing'.

    Returns
    -------
    epsilon: float
        Suppression fraction compared to R0.
    """
    CDC_MAX = 2.5  # https://www.cdc.gov/coronavirus/2019-ncov/hcp/planning-scenarios.html
    FLATTEN_THE_CURVE = 0.97
    SOCIAL_DISTANCING = 1.7

    R0 = inferred_params["R0"]
    R_eff2 = inferred_params["eps2"] * R0

    if scenario == "inferred":
        return min(R_eff2, CDC_MAX) / R0
    elif scenario == "no_intervention":
        return min(R0, CDC_MAX) / R0
    elif scenario == "flatten_the_curve":
        return FLATTEN_THE_CURVE / R0
    elif scenario == "social_distancing":
        return SOCIAL_DISTANCING / R0
    else:
        raise ValueError(f"Suppression {scenario} not valid")


def generate_triggered_suppression_model(
    t_list, lockdown_days, open_days, reduction=0.25, start_on=0
):
    """
    Generates a contact reduction model which switches a binary suppression
    policy on and off.

    Parameters
    ----------
    t_list: array-like
        List of times.
    lockdown_days: int
        Days of reduced contact rate.
    open_days:
        Days of high contact rate.
    start_on: int
        Start the lockdown fluctuation after X days.

    Returns
    -------
    suppression_model: callable
        suppression_model(t) returns the current suppression model at time t.
    """
    state = "lockdown"
    state_switch = start_on + lockdown_days
    rho = []

    if lockdown_days == 0:
        rho = np.ones(len(t_list))
    elif open_days == 0:
        rho = np.ones(len(t_list)) * reduction
    else:
        for t in t_list:

            if t >= state_switch:
                if state == "open":
                    state = "lockdown"
                    state_switch += lockdown_days
                elif state == "lockdown":
                    state = "open"
                    state_switch += open_days
            if state == "open":
                rho.append(1)
            elif state == "lockdown":
                rho.append(reduction)
    rho = np.array(rho)
    rho[t_list < start_on] = 1
    return interp1d(t_list, rho, fill_value="extrapolate")


def generate_covidactnow_scenarios(t_list, R0, t0, scenario):
    """
    Generate a suppression policy for CovidActNow which generates an Reff on a
    given date according to the policies in place.
    
    Implements CovidActNow's version, which sets Reff
        ```
        def get_interventions(start_date=datetime.now().date()):
            return [
                None,  # No Intervention
                {  # Flatten the Curve
                    start_date: 1.3,
                    start_date + timedelta(days=30) : 1.1,
                    start_date + timedelta(days=60) : 0.8,
                    start_date + timedelta(days=90) : None
                },
                {  # Full Containment
                    start_date : 1.3,
                    start_date + timedelta(days=7) : 0.3,
                    start_date + timedelta(days=30 + 7) : 0.2,
                    start_date + timedelta(days=30 + 2*7) : 0.1,
                    start_date + timedelta(days=30 + 3*7) : 0.035,
                    start_date + timedelta(days=30 + 4*7) : 0
                },
                {  # Social Distancing
                    start_date: 1.7,
                    start_date + timedelta(days=90) : None
                },
            ]
        ```

    Returns
    -------
    suppression_model: callable
        suppression_model(t) returns the current suppression model at time t.
    """
    # Rho is the multiplier on the contact rate. i.e. 1 = no change, 0 = no transmission
    rho = []
    for t in t_list:
        actual_date = t0 + timedelta(days=t)
        today = datetime.utcnow()

        if scenario == "no_intervention":
            rho.append(1)

        elif scenario == "flatten_the_curve":
            if actual_date <= today:
                rho.append(1)
            elif (actual_date - today).days <= 30:
                rho.append(0.97 / R0)
            elif (actual_date - today).days <= 60:
                rho.append(0.97 / R0)
            elif (actual_date - today).days <= 90:
                rho.append(0.97 / R0)
            else:  # Open back up...
                rho.append(1)

        elif scenario == "full_containment":
            if actual_date <= today:
                rho.append(1)
            elif (actual_date - today).days <= 7:
                rho.append(1.3 / R0)
            elif (actual_date - today).days <= 30 + 7 * 1:
                rho.append(0.3 / R0)
            elif (actual_date - today).days <= 30 + 7 * 2:
                rho.append(0.2 / R0)
            elif (actual_date - today).days <= 30 + 7 * 3:
                rho.append(0.1 / R0)
            elif (actual_date - today).days <= 30 + 7 * 4:
                rho.append(0.035 / R0)
            else:
                rho.append(0)

        elif scenario == "social_distancing":
            if actual_date <= today:
                rho.append(1)
            elif (actual_date - today).days <= 90:
                rho.append(1.7 / R0)
            else:
                rho.append(1)
        else:
            raise ValueError(f"Invalid scenario {scenario}")

    return interp1d(t_list, rho, fill_value="extrapolate")


def get_epsilon_interpolator(
    eps, t_break, eps2=-1, t_delta_phases=-1, transition_time=14, t_break_final=None, eps_final=None
):
    """
    Return an interpolator that produces an epsilon when called with a time (relative to the model
    start). The solution has at least 3 steps (4 if t_break_final and eps_final are set) with a
    linear transition between each step.

    Parameters
    ----------
    eps: float
        Suppression level after t_break
    t_break: float
        Time since simulation start to place a break.
    eps2: float
        Relative fraction of R0 for third phase
    t_delta_phases: float
        Time between first and second phase transitions
    transition_time: float
        Length of time to transition between epsilon states.
    t_break_final: int or NoneType
        Time since simulation start to place a break to a final level.
    eps_final: float
        Suppression level after t_break_final. If None, then eps is used.

    Returns
    -------
    suppression_model: callable
        suppression_model(t) returns the current suppression model at time t.
    """
    TIMEBOUNDARY = 100000

    # List of points (time, epsilon value) where transitions will occur
    points = [
        (0, 1),
        (t_break, 1),
        (t_break + transition_time, eps),
        (t_break + transition_time + t_delta_phases, eps),
        (t_break + transition_time + t_delta_phases + transition_time, eps2),
    ]

    # For estimating future trajectories we can either use the last fitted value or pass in
    # another value to the SEIR model.

    if eps_final is None:  # Use the current fit value
        points.extend([(TIMEBOUNDARY, eps2)])
    else:  # Transition to the provided value
        points.extend(
            [
                (t_break_final, eps2),
                (t_break_final + transition_time, eps_final),
                (TIMEBOUNDARY, eps_final),
            ]
        )

    x, y = zip(*points)
    return interp1d(x=x, y=y, fill_value="extrapolate")


def generate_empirical_distancing_policy(
    t_list, fips, future_suppression, reference_start_date=None
):
    """
    Produce a suppression policy based on Imperial College estimates of social
    distancing programs combined with County level datasets about their
    implementation.

    Parameters
    ----------
    t_list: array-like
        List of times to interpolate over.
    fips: str
        County fips to lookup interventions against.
    future_suppression: float
        The suppression level to apply in an ongoing basis after today, and
        going backward as the lockdown / stay-at-home efficacy.
    reference_start_date: pd.Timestamp
        Start date as reference to shift t_list.

    Returns
    -------
    suppression_model: callable
        suppression_model(t) returns the current suppression model at time t.
    """

    t0 = infer_t0(fips)
    reference_start_date = reference_start_date or t0

    rho = []

    # Check for fips that don't match.
    public_implementations = load_data.load_public_implementations_data()

    # Not all counties present in this dataset.
    if fips not in public_implementations.index:
        # Then assume 1.0 until today and then future_suppression going forward.
        for t_step in t_list:
            t_actual = t0 + timedelta(days=t_step)
            if t_actual <= datetime.now():
                rho.append(1.0)
            else:
                rho.append(future_suppression)
    else:
        policies = public_implementations.loc[fips].to_dict()
        for t_step in t_list:
            t_actual = t0 + timedelta(days=t_step)
            rho_this_t = 1

            # If this is a future date, assume lockdown continues.
            if t_actual > datetime.utcnow():
                rho.append(future_suppression)
                continue

            # If the policy was enacted on this timestep then activate it in
            # addition to others. These measures are additive unless lockdown is
            # instituted.
            for independent_measure in [
                "public_schools",
                "entertainment_gym",
                "restaurant_dine-in",
                "federal_guidelines",
            ]:

                if (
                    not pd.isnull(policies[independent_measure])
                    and t_actual > policies[independent_measure]
                ):
                    rho_this_t -= distancing_measure_suppression[independent_measure]

            # Only take the max of these, since 500 doesn't matter if 50 is enacted.
            if not pd.isnull(policies["50_gatherings"]) and t_actual > policies["50_gatherings"]:
                rho_this_t -= distancing_measure_suppression["50_gatherings"]
            elif (
                not pd.isnull(policies["500_gatherings"]) and t_actual > policies["500_gatherings"]
            ):
                rho_this_t -= distancing_measure_suppression["500_gatherings"]

            # If lockdown, then we don't care about any others, just set to
            # future suppression.
            if pd.isnull(policies["stay_at_home"]) and t_actual > policies["stay_at_home"]:
                rho_this_t = future_suppression
            rho.append(rho_this_t)

    t_list_since_reference_date = (
        t_list + (pd.to_datetime(t0) - pd.to_datetime(reference_start_date)).days
    )

    return interp1d(t_list_since_reference_date, rho, fill_value="extrapolate")


def generate_empirical_distancing_policy_by_state(
    t_list, state, future_suppression, reference_start_date=None
):
    """
    Produce a suppression policy at state level based on Imperial College
    estimates of social distancing programs combined with County level
    datasets about their implementation.

    Note: This is about 250ms per state, which adds up when running e.g. MLE
    optimization. Bottleneck is computing the suppression policy to date which
    is done by summing counties. This should be done once per state and lru
    cached, not done for each county every call. Also just using numpy instead
    of pandas.

    Parameters
    ----------
    t_list: array-like
        List of times to interpolate over.
    state: str
        State full name to lookup interventions against.
    future_suppression: float
        The suppression level to apply in an ongoing basis after today, and
        going backward as the lockdown / stay-at-home efficacy.
    reference_start_date: pd.Timestamp
        Start date as reference to shift t_list.

    Returns
    -------
    suppression_model: callable
        suppression_model(t) returns the current suppression model at time t
    """
    county_metadata = load_data.load_county_metadata()
    counties_fips = county_metadata[county_metadata.state == state].fips.unique()

    if reference_start_date is None:
        reference_start_date = min([infer_t0(fips) for fips in counties_fips])

    # Aggregate the counties to the state level, weighted by population.
    weight = county_metadata.loc[county_metadata.state == state, "total_population"].values
    weight = weight / weight.sum()
    results = []
    for fips in counties_fips:
        suppression_policy = generate_empirical_distancing_policy(
            fips=fips,
            t_list=t_list,
            future_suppression=future_suppression,
            reference_start_date=reference_start_date,
        )
        results.append(suppression_policy(t_list).clip(max=1, min=0))
    results_for_state = (np.vstack(results).T * weight).sum(axis=1)

    return interp1d(t_list, results_for_state, fill_value="extrapolate")


def piecewise_parametric_policy(x, t_list):
    """
    Generate a piecewise suppression policy over n_days based on interval
    splits at levels passed and according to the split_power_law.

    Parameters
    ----------
    x: array(float)
        x[0]: split_power_law
            The splits are generated based on relative proportions of
            t ** split_power_law. Hence split_power_law = 0 is evenly spaced.
        x[1:]: suppression_levels: array-like
            Series of suppression levels that will be equally strewn across.
    t_list: array-like
        List of days over which the period.

    Returns
    -------
    policy: callable
        Interpolator for the suppression policy.
    """
    split_power_law = x[0]
    suppression_levels = x[1:]
    period = int(np.max(t_list) - np.min(t_list))
    periods = np.array([(t + 1) ** split_power_law for t in range(len(suppression_levels))])
    periods = (periods / periods.sum() * period).cumsum()
    periods[-1] += 0.001  # Prevents floating point errors.
    suppression_levels = [suppression_levels[np.argwhere(t <= periods)[0][0]] for t in t_list]
    policy = interp1d(t_list, suppression_levels, fill_value="extrapolate")
    return policy


def fourier_parametric_policy(x, t_list, suppression_bounds=(0.5, 1.5)):
    """
    Generate a piecewise suppression policy over n_days based on interval
    splits at levels passed and according to the split_power_law.

    Parameters
    ----------
    x: array(float)
        First N coefficients for a Fourier series which becomes inversely
        transformed to generate a real series. a_0 is taken relative to level
        the mean at 0.75 (a0 = 3 * period / 4) * X[0]
    t_list: array-like
        List of days over which the period.
    suppression_bounds: tuple(float)
        Lower and upper bounds on the suppression level. This clips the fourier
        policy.

    Returns
    -------
    policy: callable
        Interpolator for the suppression policy.
    """
    frequency_domain = np.zeros(len(t_list))
    frequency_domain[0] = (3 * (t_list.max() - t_list.min()) / 4) * x[0]
    frequency_domain[1 : len(x)] = x[1:]
    time_domain = np.fft.ifft(frequency_domain).real + np.fft.ifft(frequency_domain).imag

    return interp1d(
        t_list,
        time_domain.clip(min=suppression_bounds[0], max=suppression_bounds[1]),
        fill_value="extrapolate",
    )
