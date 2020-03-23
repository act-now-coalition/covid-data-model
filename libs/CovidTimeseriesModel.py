import logging
import pandas as pd
import datetime


class CovidTimeseriesModel:
    # Initializer / Instance Attributes
    def __init__(self):
        logging.basicConfig(level=logging.CRITICAL)

    def calculate_r(self, current_cycle, previous_cycle, r0):
        # Calculate the r0 value based on the current and past number of confirmed cases
        if current_cycle['cases'] is not None:
            if previous_cycle['cases'] > 0:
                return current_cycle['cases'] / previous_cycle['cases']
        return r0

    def calculate_effective_r(self, current_cycle, previous_cycle, pop):
        # Calculate effective r by accounting for herd-immunity
        return current_cycle['r'] * (previous_cycle['ending_susceptible'] / pop)

    def calculate_newly_infected(self, current_cycle, previous_cycle, pop, initial_hospitalization_rate):
        if previous_cycle['newly_infected'] > 0:
            # If we have previously known cases, use the R0 to estimate newly infected cases.
            newly_infected = previous_cycle['newly_infected'] * self.calculate_effective_r(current_cycle,
                                                                                              previous_cycle, pop)
        else:
            # TODO: Review. I'm having trouble following this block
            # We assume the first positive cases were exclusively hospitalized ones.
            actual_infected_vs_tested_positive = 1 / initial_hospitalization_rate  # ~20
            newly_infected = current_cycle['cases'] * actual_infected_vs_tested_positive
        return newly_infected

    def calculate_currently_infected(self, cycle_series, rolling_intervals_for_current_infected):
        # Calculate the number of people who have been infected but are no longer infected (one way or another)
        # Better way:
        return sum(
            ss['newly_infected'] for ss in cycle_series[-rolling_intervals_for_current_infected:]
        )

    def hospital_is_overloaded(self, current_cycle, previous_cycle):
        # Determine if a hospital is overloaded. Trivial now, but will likely become more complicated in the near future
        return previous_cycle['available_hospital_beds'] > current_cycle['predicted_hospitalized']

    def calculate_cumulative_deaths(self, current_cycle, previous_cycle, case_fatality_rate,
                                    case_fatality_rate_hospitals_overwhelmed):
        # If the number of hospital beds available is exceeded by the number of patients that need hospitalization,
        #  the death rate increases
        # Can be significantly improved in the future
        if self.hospital_is_overloaded(current_cycle, previous_cycle):
            return previous_cycle['cumulative_deaths'] + round(current_cycle['newly_infected'] * case_fatality_rate)
        else:
            return previous_cycle['cumulative_deaths'] + round(
                current_cycle['newly_infected'] * (case_fatality_rate + case_fatality_rate_hospitals_overwhelmed)
            )

    def calcluate_recovered_or_died(self, current_cycle, previous_cycle, cycle_series,
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

    def calculate_ending_susceptible(self, current_cycle, previous_cycle, pop):
        # Number of people who haven't been sick yet
        return round(
            pop - (current_cycle['newly_infected'] + current_cycle['currently_infected'] + current_cycle['recovered_or_died'])
        )

    def calculate_estimated_actual_chance_of_infection(self, current_cycle, previous_cycle, hospitalization_rate, pop):
        # Reflects the model logic, but probably needs to be changed
        if current_cycle['cases'] is not None:
            return ((current_cycle['cases'] / hospitalization_rate) * 2) / pop
        return None

    def calculate_actual_reported(self, current_cycle, previous_cycle):
        # Needs to account for creating synthetic data for missing records
        if current_cycle['cases'] is not None:
            return round(current_cycle['cases'])
        return 0

    def calculate_available_hospital_beds(self, current_cycle, previous_cycle, max_hospital_capacity_factor,
                                          original_available_hospital_beds, hospital_capacity_change_daily_rate):
        available_hospital_beds = previous_cycle['available_hospital_beds']
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
                # Calculate the date of this current_cycle. Should be the initial date, plus one interval for every entry
                # in the timeseries list, plus one for the initial current_cycle
                'date': current_cycle_date,
                'cases': None,
                'deaths': None,
                'recovered': None
            }

    def iterate_model(self, iterations, model_parameters):
        """
        The guts. Creates the initial conditions, then iterates the model over the data for a specified number
        of iterations
        """
        # Get the earliest date in the data
        model_parameters['init_date'] = model_parameters['timeseries'].sort_values('date')['date'].iloc[0]
        original_available_hospital_beds = round(model_parameters['beds'] * (1 - model_parameters['initial_hospital_bed_utilization']), 0)

        # Prepare the initial conditions for the loop
        # Initialize the series with the init current_cycle
        cycle_series = [
            {
                # We want the initial cycle to be one interval behind the first iteration of data we have
                'date': model_parameters['init_date'] - datetime.timedelta(days=model_parameters['model_interval']),
                'r': model_parameters['r0'],
                'cases': 0,
                'actual_reported': 0,
                'current_infected': 0,
                'est_actual_chance_of_infection': 0,
                'newly_infected': 0,
                'currently_infected': 0,
                'cumulative_infected': 0,
                'cumulative_deaths': 0,
                'recovered_or_died': 0,
                'ending_susceptible': model_parameters['population'],
                'predicted_hospitalized': 0,
                'available_hospital_beds': original_available_hospital_beds
            },
        ]
        previous_cycle = cycle_series[0]
        for i in range(0, len(model_parameters['timeseries']) + iterations - 1):
            # Step through existing empirical data
            current_cycle = self.make_cycle(i, model_parameters)
            logging.debug('Calculating values for {}'.format(current_cycle['date']))

            # Calculate the r0 value
            current_cycle['r'] = self.calculate_r(current_cycle, previous_cycle, model_parameters['r0'])
            # Calculate the number of newly infected cases
            current_cycle['newly_infected'] = self.calculate_newly_infected(
                current_cycle,
                previous_cycle,
                model_parameters['population'],
                model_parameters['initial_hospitalization_rate']
            )
            # Assume infected cases from before the rolling interval have concluded.
            current_cycle['recovered_or_died'] = self.calcluate_recovered_or_died(
                current_cycle,
                previous_cycle,
                cycle_series,
                model_parameters['rolling_intervals_for_current_infected']
            )
            # Calculate the number of people who have already been infected
            current_cycle['currently_infected'] = self.calculate_currently_infected(
                cycle_series,
                model_parameters['rolling_intervals_for_current_infected']
            )
            # Calculate the cumulative number of infected individuals
            current_cycle['cumulative_infected'] = previous_cycle['cumulative_infected'] + current_cycle['newly_infected']
            # Predict the number of patients that will require hospitilzation
            current_cycle['predicted_hospitalized'] = current_cycle['newly_infected'] * model_parameters['hospitalization_rate']
            # Calculate the number of cumulative deaths
            current_cycle['cumulative_deaths'] = self.calculate_cumulative_deaths(
                current_cycle,
                previous_cycle,
                model_parameters['case_fatality_rate'],
                model_parameters['case_fatality_rate_hospitals_overwhelmed'],
            )
            # Recalculate the estimated chance of infection
            current_cycle['est_actual_chance_of_infection'] = self.calculate_estimated_actual_chance_of_infection(
                current_cycle,
                previous_cycle,
                model_parameters['hospitalization_rate'],
                model_parameters['population']
            )
            # Note the actual number of reported cases
            current_cycle['actual_reported'] = self.calculate_actual_reported(current_cycle, previous_cycle)
            # Recalculate how many people are susceptible at the end of the current_cycle
            current_cycle['ending_susceptible'] = self.calculate_ending_susceptible(current_cycle, previous_cycle,
                                                                               model_parameters['population'])
            # Recalculate how many hospital beds are left
            current_cycle['available_hospital_beds'] = self.calculate_available_hospital_beds(
                current_cycle,
                previous_cycle,
                model_parameters['max_hospital_capacity_factor'],
                original_available_hospital_beds,
                model_parameters['hospital_capacity_change_daily_rate']
            )
            # Prepare for the next iteration
            cycle_series.append(current_cycle)
            previous_cycle = current_cycle
        return cycle_series

    def forecast_region(self, iterations, model_parameters):
        cycle_series = self.iterate_model(iterations, model_parameters)

        return pd.DataFrame({
            'Note': ['' for s in cycle_series],
            'Date': [s['date'] for s in cycle_series],
            'Timestamp': [
                # Create a UNIX timestamp for each datetime. Easier for graphs to digest down the road
                datetime.datetime(year=s['date'].year, month=s['date'].month, day=s['date'].day).timestamp()
                for s in cycle_series
            ],
            'Eff. R0': [s['r'] for s in cycle_series],
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
