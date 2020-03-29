import logging
import math
import pandas as pd
import datetime

_logger = logging.getLogger(__name__)


class CovidTimeseriesModel:
    # Initializer / Instance Attributes

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
        if previous_cycle['newly_infected'] > 0:
            return previous_cycle['newly_infected'] * current_cycle['effective_r']
        if current_cycle['cases'] is not None:
            return current_cycle['newly_infected'] / model_parameters['hospitalization_rate']
        return 0

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

    def make_cycle(self, i, model_parameters):
        # Check if the date we are trying to model is within the data we have
        current_cycle_date = model_parameters['init_date'] + datetime.timedelta(days=model_parameters['model_interval'] * i)
        dates = model_parameters['timeseries']['date']
        if any(dates.isin([current_cycle_date])):
            # Grab the cycle corresponding to this iteration
            # The data comes in as a DataFrame, but within the model we use a dict. We can revisit that decision
            # First we need to sort the list by date  to ensure we're getting the correct one
            return model_parameters['timeseries'][model_parameters['timeseries']['date'] == current_cycle_date].to_dict('records')[0]
        else:
            # If we have exceeded the number of rows in the collected data, we go into projection mode.
            return {
                'date': current_cycle_date,
                'cases': None,
                'deaths': None,
                'recovered': None
            }

    def build_init_cycle(self, model_parameters, init_data_cycle):
        # There are some assumptions made when we create the fisrt data cycle. This encapsulates all of those
        #  assumptions into one place
        init_cycle = {
            # We want the initial cycle to be one interval behind the first iteration of data we have
            'date': model_parameters['init_date'],
            'r': model_parameters['r0'],
            'effective_r': None,
            'cases': init_data_cycle['cases'],
            'actual_reported': init_data_cycle['cases'],
            'current_infected': 0,
            'est_actual_chance_of_infection': 0,
            'newly_infected': init_data_cycle['cases'] / model_parameters['hospitalization_rate'],
            'currently_infected': 0,
            'cumulative_infected': None,
            'cumulative_deaths': None,
            'recovered_or_died': 0,
            'ending_susceptible': model_parameters['population'] - (init_data_cycle['cases'] / model_parameters['hospitalization_rate']),
            'predicted_hospitalized': 0,
            'available_hospital_beds': model_parameters['original_available_hospital_beds']
        }
        return init_cycle

    def initialize_parameters(self, model_parameters):
        """Perform all of the necessary setup prior to the model's execution"""
        # Get the earliest date in the data
        init_data_cycle = model_parameters['timeseries'].sort_values('date').iloc[0]
        model_parameters['init_date'] = init_data_cycle['date']
        # Get the latest date in the data
        model_parameters['last_date'] = model_parameters['timeseries'].sort_values('date')['date'].iloc[-1]
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
        # Prepare the initial conditions for the loop
        # Return the start of the cycle_series object with the initial cycle contained in it
        return [self.build_init_cycle(model_parameters, init_data_cycle),], model_parameters

    def iterate_model(self, model_parameters):
        """The guts. Creates the initial conditions, iterates the model over the data for a specified number
        of iterations, collects the results, and returns them"""
        cycle_series, model_parameters = self.initialize_parameters(model_parameters)
        previous_cycle = cycle_series[0]
        for i in range(1, model_parameters['total_iterations']):
            # Step through existing empirical data
            current_cycle = self.make_cycle(i, model_parameters)
            current_cycle['i'] = i  # Note the cycle
            _logger.debug('Calculating values for {}'.format(current_cycle['date']))
            # Calculate the r0 value
            r = self.calculate_r(current_cycle, previous_cycle, model_parameters)
            current_cycle['r'] = r
            eff_r = self.calculate_effective_r(current_cycle, previous_cycle, model_parameters)
            current_cycle['effective_r'] = eff_r
            # Calculate the number of newly infected cases
            ni = self.calculate_newly_infected(
                current_cycle,
                previous_cycle,
                model_parameters
            )
            current_cycle['newly_infected'] = ni
            ci = self.calculate_cumulative_infected(current_cycle, previous_cycle)
            # Calculate the cumulative number of infected individuals
            current_cycle['cumulative_infected'] = ci
            # Assume infected cases from before the rolling interval have concluded.
            curr_i = self.calculate_currently_infected(
                            cycle_series,
                            model_parameters['rolling_intervals_for_current_infected']
                        )
            current_cycle['currently_infected'] = curr_i
            # Calculate the number of people who have recovered or died
            rod = self.calculuate_recovered_or_died(
                current_cycle,
                previous_cycle,
                cycle_series,
                model_parameters['rolling_intervals_for_current_infected']
            )
            current_cycle['recovered_or_died'] = rod
            # Predict the number of patients that will require hospitilzation
            ph = self.calculate_predicted_hospitalized(current_cycle,
                                                       model_parameters)
            current_cycle['predicted_hospitalized'] = ph
            # Calculate the number of cumulative deaths
            cum_d = self.calculate_cumulative_deaths(
                current_cycle,
                previous_cycle,
                model_parameters['case_fatality_rate'],
                model_parameters['case_fatality_rate_hospitals_overwhelmed'],
            )
            current_cycle['cumulative_deaths'] = cum_d
            # Recalculate the estimated chance of infection
            est_chance_inf = self.calculate_estimated_actual_chance_of_infection(
                current_cycle,
                previous_cycle,
                model_parameters['hospitalization_rate'],
                model_parameters['population']
            )
            current_cycle['est_actual_chance_of_infection'] = est_chance_inf
            # Note the actual number of reported cases
            ar = self.calculate_actual_reported(current_cycle, previous_cycle)
            current_cycle['actual_reported'] = ar
            # Recalculate how many people are suscep|tible at the end of the current_cycle
            es = self.calculate_ending_susceptible(current_cycle, previous_cycle,
                                                   model_parameters)
            current_cycle['ending_susceptible'] = es
            # Recalculate how many hospital beds are left
            ahb = self.calculate_available_hospital_beds(
                current_cycle,
                previous_cycle,
                model_parameters['max_hospital_capacity_factor'],
                model_parameters['original_available_hospital_beds'],
                model_parameters['hospital_capacity_change_daily_rate']
            )
            current_cycle['available_hospital_beds'] = ahb
            # Prepare for the next iteration
            cycle_series.append(current_cycle)
            previous_cycle = current_cycle
        return cycle_series

    def forecast(self, model_parameters):
        cycle_series = self.iterate_model(model_parameters)
        return pd.DataFrame({
            'Note': ['' for s in cycle_series],
            'Date': [s['date'] for s in cycle_series],
            'Timestamp': [
                # Create a UNIX timestamp for each datetime. Easier for graphs to digest down the road
                datetime.datetime(year=s['date'].year, month=s['date'].month, day=s['date'].day).timestamp()
                for s in cycle_series
            ],
            'R': [s['r'] for s in cycle_series],
            'Effective R.': [s['effective_r'] for s in cycle_series],
            'Beg. Susceptible': [s['ending_susceptible'] for s in cycle_series],
            'New Inf.': [s['newly_infected'] for s in cycle_series],
            'Curr. Inf.': [s['currently_infected'] for s in cycle_series],
            'Recov. or Died': [s['recovered_or_died'] for s in cycle_series],
            'End Susceptible': [s['ending_susceptible'] for s in cycle_series],
            'Actual Reported': [s['actual_reported'] for s in cycle_series],
            'Pred. Hosp.': [s['predicted_hospitalized'] for s in cycle_series],
            'Cum. Inf.': [s['cumulative_infected'] for s in cycle_series],
            'Cum. Deaths': [s['cumulative_deaths'] for s in cycle_series],
            'Avail. Hosp. Beds': [s['available_hospital_beds'] for s in cycle_series],
            'S&P 500': [None for s in cycle_series],
            'Est. Actual Chance of Inf.': [s['est_actual_chance_of_infection'] for s in cycle_series],
            'Pred. Chance of Inf.': [None for s in cycle_series],
            'Cum. Pred. Chance of Inf.': [None for s in cycle_series],
            'R0': [None for s in cycle_series],
            '% Susceptible': [None for s in cycle_series]
        })
