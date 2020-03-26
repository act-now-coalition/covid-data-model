import logging
import math

import numpy as np
import pandas as pd
import datetime

import pprint

from scipy.integrate import odeint


# @TODO: Switch to this model.
class CovidTimeseriesCycle(object):
    def __init__(self, model_parameters, init_data_cycle):
        # There are some assumptions made when we create the fisrt data cycle. This encapsulates all of those
        #  assumptions into one place

        self.date = model_parameters['init_date']
        self.r0 = model_parameters['r0']
        self.effective_r0 = None
        self.cases = init_data_cycle['cases']
        self.actual_reported = init_data_cycle['cases']
        self.current_infected = 0
        self.newly_infected_from_confirmed = init_data_cycle['cases'] * model_parameters['estimated_new_cases_per_confirmed']
        self.newly_infected_from_deaths = 0
        self.newly_infected = self.newly_infected_from_confirmed + self.newly_infected_from_deaths
        self.currently_infected = 0
        self.cumulative_infected = None
        self.cumulative_deaths = None
        self.recovered_or_died = 0
        self.ending_susceptible = model_parameters['population'] - (init_data_cycle['cases'] / model_parameters['hospitalization_rate'])
        self.predicted_hospitalized = 0
        self.available_hospital_beds = model_parameters['original_available_hospital_beds']

    def __iter__(self):
        return self

    def __next__(self):
        pass

class CovidTimeseriesModelSIR:
    # Initializer / Instance Attributes
    def __init__(self):
        logging.basicConfig(level=logging.ERROR)

    def calculate_r(self, current_cycle, previous_cycle, model_parameters):
        # Calculate the r0 value based on the current and past number of confirmed cases
        if current_cycle['cases'] is not None:
            if previous_cycle['cases'] > 0:
                return current_cycle['cases'] / previous_cycle['cases']
        return model_parameters['r0']

    def calculate_effective_r(self, current_cycle, previous_cycle, model_parameters):
        # In order to account for interventions, we need to allow effective r to be changed manually.
        if model_parameters['interventions'] is not None:
            # The interventions come in as a dict, with date's as keys and effective r's as values
            # Find the most recent intervention we have passed
            for d in sorted(model_parameters['interventions'].keys())[::-1]:
                if current_cycle['date'] >= d:  # If the current cycle occurs on or after an intervention
                    return model_parameters['interventions'][d] * (previous_cycle['ending_susceptible'] / model_parameters['population'])
        # Calculate effective r by accounting for herd-immunity
        return current_cycle['r'] * (previous_cycle['ending_susceptible'] / model_parameters['population'])

    def calculate_newly_infected(self, current_cycle, previous_cycle, model_parameters):
        # If we have previously known cases, use the R0 to estimate newly infected cases.
        nic = 0
        if previous_cycle['newly_infected'] > 0:
            nic = previous_cycle['newly_infected'] * current_cycle['effective_r']
        elif current_cycle['cases'] is not None:
            if not 'newly_infected' in current_cycle:
                pprint.pprint(current_cycle)
            nic = current_cycle['newly_infected'] * model_parameters['estimated_new_cases_per_confirmed']

        nid = 0
        if current_cycle['deaths'] is not None:
            nid = current_cycle['deaths'] * model_parameters['estimated_new_cases_per_death']

        return (nic, nid)

    def calculate_currently_infected(self, cycle_series, rolling_intervals_for_current_infected):
        # Calculate the number of people who have been infected but are no longer infected (one way or another)
        # Better way:
        return sum(
            ss['newly_infected'] for ss in cycle_series[-rolling_intervals_for_current_infected:]
        )

    def hospital_is_overloaded(self, current_cycle, previous_cycle):
        # Determine if a hospital is overloaded. Trivial now, but will likely become more complicated in the near future
        return previous_cycle['available_hospital_beds'] < current_cycle['predicted_hospitalized']

    def calculate_cumulative_infected(self, current_cycle, previous_cycle):
        if previous_cycle['cumulative_infected'] is None:
            return current_cycle['newly_infected']
        return previous_cycle['cumulative_infected'] + current_cycle['newly_infected']

    def calculate_cumulative_deaths(self, current_cycle, previous_cycle, case_fatality_rate,
                                    case_fatality_rate_hospitals_overwhelmed):
        # If the number of hospital beds available is exceeded by the number of patients that need hospitalization,
        #  the death rate increases
        # Can be significantly improved in the future
        if previous_cycle['cumulative_deaths'] is None:
            return current_cycle['cumulative_infected'] * case_fatality_rate
        if not self.hospital_is_overloaded(current_cycle, previous_cycle):
            return previous_cycle['cumulative_deaths'] + (current_cycle['newly_infected'] * case_fatality_rate)
        else:
            return previous_cycle['cumulative_deaths'] + (
                current_cycle['newly_infected'] * (case_fatality_rate + case_fatality_rate_hospitals_overwhelmed)
            )

    def calculuate_recovered_or_died(self, current_cycle, previous_cycle, cycle_series,
                                    rolling_intervals_for_current_infected):
        # Recovered or died (RoD) is a cumulative number. We take the number of RoD from last current_cycle, and add to it
        #  the number of individuals who were newly infected a set number of current_cycles ago (rolling_intervals_for_current_infected)
        #  (RICI). It is assumed that after the RICI, an infected interval has either recovered or died.
        # Can be improved in the future
        if len(cycle_series) >= (rolling_intervals_for_current_infected + 1):
            return previous_cycle['recovered_or_died'] + \
                   cycle_series[-(rolling_intervals_for_current_infected + 1)]['newly_infected']
        else:
            return previous_cycle['recovered_or_died']

    def calculate_predicted_hospitalized(self, current_cycle, model_parameters):
        return current_cycle['newly_infected'] * \
               model_parameters['hospitalization_rate']

    def calculate_ending_susceptible(self, current_cycle, previous_cycle, model_parameters):
        # Number of people who haven't been sick yet
        return model_parameters['population'] - (current_cycle['newly_infected'] + current_cycle['currently_infected'] + current_cycle['recovered_or_died'])

    def calculate_estimated_actual_chance_of_infection(self, current_cycle, previous_cycle, hospitalization_rate, pop):
        # Reflects the model logic, but probably needs to be changed
        if current_cycle['cases'] is not None:
            return ((current_cycle['cases'] / hospitalization_rate) * 2) / pop
        return None

    def calculate_actual_reported(self, current_cycle, previous_cycle):
        # Needs to account for creating synthetic data for missing records
        if current_cycle['cases'] is not None:
            return current_cycle['cases']
        return None

    def calculate_available_hospital_beds(self, current_cycle, previous_cycle, max_hospital_capacity_factor,
                                          original_available_hospital_beds, hospital_capacity_change_daily_rate):
        available_hospital_beds = previous_cycle['available_hospital_beds']
        if current_cycle['i'] < 3:
            return available_hospital_beds
        if available_hospital_beds < max_hospital_capacity_factor * original_available_hospital_beds:
            # Hospitals can try to increase their capacity for the sick
            available_hospital_beds *= hospital_capacity_change_daily_rate
        return available_hospital_beds

    def initialize_parameters(self, model_parameters):
        """Perform all of the necessary setup prior to the model's execution"""
        # Get the earliest date in the data where there is at least 1 infection
        init_data_cycle = model_parameters['timeseries'].sort_values('date')

        init_data_cycle = init_data_cycle.loc[(init_data_cycle['active'] > 0), :].iloc[0]

        model_parameters['init_date'] = init_data_cycle['date']

        # Get the last day of the model based on the number of iterations and
        # length of the iterations
        duration = datetime.timedelta(days=(model_parameters['model_interval'] * model_parameters['projection_iterations']))
        model_parameters['last_date'] = model_parameters['init_date'] + duration

        # Calculate the number of iterations needed to cover the data
        model_parameters['data_iterations'] = math.floor(
            # timedelta subtraction is not inclusive, but we need it to be, so we'll add one day
            (model_parameters['last_date'] - (model_parameters['init_date'] - datetime.timedelta(days=1))) /
            datetime.timedelta(days=model_parameters['model_interval'])
        )
        # Sum the data and projection interations for a single number of iterations to cycle the model
        model_parameters['total_iterations'] = model_parameters['data_iterations'] + model_parameters['projection_iterations']
        model_parameters['original_available_hospital_beds'] = \
            model_parameters['beds'] * (1 - model_parameters['initial_hospital_bed_utilization'])

        return model_parameters

    # The SIR model differential equations.
    # we'll blow these out to add E, D, and variations on I (to SIR)
    # but these are the basics
    # y = initial conditions
    # t = a grid of time points (in days) - not currently used, but will be for time-dependent functions
    # N = total pop
    # beta = contact rate
    # gamma = mean recovery rate
    def deriv(self, y, t, N, beta, gamma):
        t = None

        S, I, R = y
        dSdt = -beta * S * I / N
        dIdt = beta * S * I / N - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    # Sets up and runs the integration
    # start date and end date give the bounds of the simulation
    # pop_dict contains the initial populations
    # beta = contact rate
    # gamma = mean recovery rate
    def sir(self, start_date, end_date, pop_dict, beta, gamma):
        susceptible = pop_dict['total'] - pop_dict['infected'] - pop_dict['recovered'] - pop_dict['deaths']

        y0 = susceptible, pop_dict['infected'], pop_dict['recovered']

        delta = end_date - start_date

        t = np.linspace(0, delta.days, delta.days )

        ret = odeint(self.deriv, y0, t, args=(pop_dict['total'], beta, gamma))

        return ret.T

    def iterate_model(self, model_parameters):
        """The guts. Creates the initial conditions, and runs the SIR model for the
        specified number of iterations with the given inputs"""
        model_parameters = self.initialize_parameters(model_parameters)

        timeseries = model_parameters['timeseries'].sort_values('date')

        timeseries.loc[:, ['cases', 'deaths', 'recovered', 'active']] = \
            timeseries.loc[:, ['cases', 'deaths', 'recovered', 'active']].fillna(0)

        timeseries['dt'] = pd.to_datetime(timeseries['date']).dt.date
        timeseries.set_index('dt', inplace=True)
        timeseries.sort_index(inplace=True)

        init_date = model_parameters['init_date'].to_pydatetime().date()

        # load the initial populations
        pop_dict = {
            'total': timeseries.loc[init_date, 'population'],
            'infected': timeseries.loc[init_date, 'active'],
            'recovered': timeseries.loc[init_date, 'recovered'],
            'deaths': timeseries.loc[init_date, 'deaths']
        }

        (S, I, R) = self.sir(model_parameters['init_date'], model_parameters['last_date'],
                             pop_dict, model_parameters['r0'], 1 / model_parameters['total_infected_period'])

        dates = pd.date_range(start=model_parameters['init_date'],
                              end=(model_parameters['last_date'] - datetime.timedelta(days=1)),
                              freq='D').to_list()

        sir_df = pd.DataFrame(zip(S, I, R), columns=['susceptible', 'infected', 'recovered'], index=dates)

        # this should be done, but belt and suspenders for the diffs()
        sir_df.sort_index(inplace=True)
        sir_df.index.name = 'date'
        sir_df.reset_index(inplace=True)

        sir_df['total'] = pop_dict['total']

        # set some of the paramters... I'm sure I'm misinterpreting some
        # and of course a lot of these don't move like they should for the model yet
        sir_df['r'] = model_parameters['r0']
        sir_df['effective_r'] = model_parameters['r0']
        sir_df['ending_susceptible'] = sir_df['susceptible']
        sir_df['currently_infected'] = sir_df['infected']

        # not sure about these guys... just doing new infections
        sir_df['newly_infected_from_confirmed'] = sir_df['infected'].diff()
        sir_df['newly_infected_from_deaths'] = sir_df['infected'].diff()

        #fillna
        sir_df.loc[:, ['newly_infected_from_confirmed', 'newly_infected_from_deaths']] = \
                sir_df.loc[:, ['newly_infected_from_confirmed', 'newly_infected_from_deaths']].fillna(0)

        # cumsum the diff (no D yet)
        sir_df['cumulative_infected'] = sir_df['newly_infected_from_confirmed'].cumsum()

        # no D yet in model
        sir_df['recovered_or_died'] = sir_df['recovered']

        # cumsum the diff (no D yet)
        sir_df['newly_died'] = sir_df['recovered'].diff()
        sir_df['cumulative_deaths'] = sir_df['newly_died'].cumsum()

        # have not broken out asymptomatic from hospitalized/severe yet
        sir_df['predicted_hospitalized'] = sir_df['infected']

        sir_df['newly_infected'] = sir_df['infected']

        # TODO: work on all these guys
        # 'actual_reported'
        # 'predicted_hospitalized'
        # 'cumulative_infected'
        # 'cumulative_deaths'
        # 'available_hospital_beds'
        sir_df['actual_reported'] = 0
        sir_df['predicted_hospitalized'] = 0
        sir_df['available_hospital_beds'] = 0

        return sir_df.to_dict('records') # cycle_series

    def forecast_region(self, model_parameters):
        cycle_series = self.iterate_model(model_parameters)

        return pd.DataFrame({
            'Date': [s['date'] for s in cycle_series],
            'Timestamp': [
                # Create a UNIX timestamp for each datetime. Easier for graphs to digest down the road
                datetime.datetime(year=s['date'].year, month=s['date'].month, day=s['date'].day).timestamp()
                for s in cycle_series
            ],
            'R': [s['r'] for s in cycle_series],
            'Effective R.': [s['effective_r'] for s in cycle_series],
            'Beg. Susceptible': [s['ending_susceptible'] for s in cycle_series],
            'New Inf < C': [int(round(s['newly_infected_from_confirmed'])) for s in cycle_series],
            'New Inf < D': [int(round(s['newly_infected_from_deaths'])) for s in cycle_series],
            'New Inf.': [int(round(s['newly_infected'])) for s in cycle_series],
            'Curr. Inf.': [s['currently_infected'] for s in cycle_series],
            'Recov. or Died': [s['recovered_or_died'] for s in cycle_series],
            'End Susceptible': [s['ending_susceptible'] for s in cycle_series],
            'Actual Reported': [s['actual_reported'] for s in cycle_series],
            'Pred. Hosp.': [s['predicted_hospitalized'] for s in cycle_series],
            'Cum. Inf.': [s['cumulative_infected'] for s in cycle_series],
            'Cum. Deaths': [s['cumulative_deaths'] for s in cycle_series],
            'Avail. Hosp. Beds': [s['available_hospital_beds'] for s in cycle_series]
        })
