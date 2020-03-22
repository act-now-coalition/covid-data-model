import logging
import pandas as pd
import datetime


class CovidTimeseriesModel:
    # Initializer / Instance Attributes
    def __init__(self):
        logging.basicConfig(level=logging.CRITICAL)

    def calculate_r(self, cycle, previous_cycle, r0):
        # Calculate the r0 value based on the current and past number of confirmed cases
        if cycle['cases'] is not None:
            if previous_cycle['cases'] > 0:
                return cycle['cases'] / previous_cycle['cases']
        return r0

    def calculate_effective_r(self, cycle, previous_cycle, pop):
        # Calculate effective r by accounting for herd-immunity
        return cycle['r'] * (previous_cycle['ending_susceptible'] / pop)

    def calculate_newly_infected(self, cycle, previous_cycle, pop, initial_hospitalization_rate):
        if previous_cycle['newly_infected'] > 0:
            # If we have previously known cases, use the R0 to estimate newly infected cases.
            newly_infected = previous_cycle['newly_infected'] * self.calculate_effective_r(cycle,
                                                                                              previous_cycle, pop)
        else:
            # TODO: Review. I'm having trouble following this block
            # We assume the first positive cases were exclusively hospitalized ones.
            actual_infected_vs_tested_positive = 1 / initial_hospitalization_rate  # ~20
            if cycle['cases'] is not None:
                confirmed = cycle['cases']
            newly_infected = cycle['cases'] * actual_infected_vs_tested_positive
        return newly_infected

    def calculate_currently_infected(self, cycle_series, rolling_intervals_for_current_infected):
        # Calculate the number of people who have been infected but are no longer infected (one way or another)
        # Better way:
        return sum(
            ss['newly_infected'] for ss in cycle_series[-rolling_intervals_for_current_infected:]
        )

    def hospital_is_overloaded(self, cycle, previous_cycle):
        # Determine if a hospital is overloaded. Trivial now, but will likely become more complicated in the near future
        return previous_cycle['available_hospital_beds'] > cycle['predicted_hospitalized']

    def calculate_cumulative_deaths(self, cycle, previous_cycle, case_fatality_rate,
                                    case_fatality_rate_hospitals_overwhelmed):
        # If the number of hospital beds available is exceeded by the number of patients that need hospitalization,
        #  the death rate increases
        # Can be significantly improved in the future
        if self.hospital_is_overloaded(cycle, previous_cycle):
            return previous_cycle['cumulative_deaths'] + round(cycle['newly_infected'] * case_fatality_rate)
        else:
            return previous_cycle['cumulative_deaths'] + round(
                cycle['newly_infected'] * (case_fatality_rate + case_fatality_rate_hospitals_overwhelmed)
            )

    def calcluate_recovered_or_died(self, snapsnot, previous_cycle, cycle_series,
                                    rolling_intervals_for_current_infected):
        # Recovered or died (RoD) is a cumulative number. We take the number of RoD from last cycle, and add to it
        #  the number of individuals who were newly infected a set number of cycles ago (rolling_intervals_for_current_infected)
        #  (RICI). It is assumed that after the RICI, an infected interval has either recovered or died.
        # Can be improved in the future
        if len(cycle_series) >= (rolling_intervals_for_current_infected + 1):
            return previous_cycle['recovered_or_died'] + \
                   cycle_series[-(rolling_intervals_for_current_infected + 1)]['newly_infected']
        else:
            return previous_cycle['recovered_or_died']

    def calculate_ending_susceptible(self, cycle, previous_cycle, pop):
        # Number of people who haven't been sick yet
        return round(
            pop - (cycle['newly_infected'] + cycle['currently_infected'] + cycle['recovered_or_died'])
        )

    def calculate_estimated_actual_chance_of_infection(self, cycle, previous_cycle, hospitalization_rate, pop):
        # Reflects the model logic, but probably needs to be changed
        if cycle['cases'] is not None:
            return ((cycle['cases'] / hospitalization_rate) * 2) / pop
        return None

    def calculate_actual_reported(self, cycle, previous_cycle):
        # Needs to account for creating synthetic data for missing records
        if cycle['cases'] is not None:
            return round(cycle['cases'])
        return 0

    def calculate_available_hospital_beds(self, cycle, previous_cycle, max_hospital_capacity_factor,
                                          original_available_hospital_beds, hospital_capacity_change_daily_rate):
        available_hospital_beds = previous_cycle['available_hospital_beds']
        if available_hospital_beds < max_hospital_capacity_factor * original_available_hospital_beds:
            # Hospitals can try to increase their capacity for the sick
            available_hospital_beds *= hospital_capacity_change_daily_rate
        return available_hospital_beds

    def get_cycle(self, i, model_parameters):
        if i < len(model_parameters['timeseries']):
            # Grab the cycle corresponding to this iteration
            # The data comes in as a DataFrame, but within the model we use a dict. We can revisit that decision
            # First we need to sort the list by date  to ensure we're getting the correct one
            cycle = model_parameters['timeseries'].sort_values('date').iloc[i].to_dict()
        else:
            # If we have exceeded the number of rows in the collected data, we go into projection mode.
            cycle = {
                # Calculate the date of this cycle. Should be the initial date, plus one interval for every entry
                # in the timeseries list, plus one for the initial cycle
                'date': model_parameters['init_date'] +
                        datetime.timedelta(days=model_parameters['model_interval'] * (i + 1)),
                'cases': None,
                'deaths': None,
                'recovered': None
            }
        return cycle

    def iterate_model(self, iterations, model_parameters):
        """
        The guts. Creates the initial conditions, then iterates the model over the data for a specified number
        of iterations
        """
        # Unpack the model parameters
        r0 = model_parameters['r0']
        rolling_intervals_for_current_infected = model_parameters['rolling_intervals_for_current_infected']
        case_fatality_rate = model_parameters['case_fatality_rate']
        case_fatality_rate_hospitals_overwhelmed = model_parameters['case_fatality_rate_hospitals_overwhelmed']
        initial_hospital_bed_utilization = model_parameters['initial_hospital_bed_utilization']
        initial_hospitalization_rate = model_parameters['initial_hospitalization_rate']
        hospitalization_rate = model_parameters['hospitalization_rate']
        max_hospital_capacity_factor = model_parameters['max_hospital_capacity_factor']
        hospital_capacity_change_daily_rate = model_parameters['hospital_capacity_change_daily_rate']

        # @TODO: See if today's data is already available. If so, don't subtract an additional day.
        # @TODO: Switch back to 1 after testing
        today = datetime.date.today() - datetime.timedelta(days=2)
        # Get the earliest date in the data
        model_parameters['init_date'] = model_parameters['timeseries'].sort_values('date')['date'].iloc[0]
        original_available_hospital_beds = round(model_parameters['beds'] * (1 - initial_hospital_bed_utilization), 0)

        # Prepare the initial conditions for the loop
        # Initialize the series with the init cycle
        cycle_series = [
            {
                # We want the initial cycle to be one interval behind the first iteration of data we have
                'date': model_parameters['init_date'] - datetime.timedelta(days=model_parameters['model_interval']),
                'r': r0,
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
            cycle = self.get_cycle(i, model_parameters)
            logging.debug('Calculating values for {}'.format(cycle['date']))

            # Calculate the r0 value
            cycle['r'] = self.calculate_r(cycle, previous_cycle, r0)
            # Calculate the number of newly infected cases
            cycle['newly_infected'] = self.calculate_newly_infected(
                cycle,
                previous_cycle,
                model_parameters['population'],
                initial_hospitalization_rate
            )
            # Assume infected cases from before the rolling interval have concluded.
            cycle['recovered_or_died'] = self.calcluate_recovered_or_died(
                cycle,
                previous_cycle,
                cycle_series,
                rolling_intervals_for_current_infected
            )
            # Calculate the number of people who have already been infected
            cycle['currently_infected'] = self.calculate_currently_infected(
                cycle_series,
                rolling_intervals_for_current_infected
            )
            # Calculate the cumulative number of infected individuals
            cycle['cumulative_infected'] = previous_cycle['cumulative_infected'] + cycle['newly_infected']
            # Predict the number of patients that will require hospitilzation
            cycle['predicted_hospitalized'] = cycle['newly_infected'] * hospitalization_rate
            # Calculate the number of cumulative deaths
            cycle['cumulative_deaths'] = self.calculate_cumulative_deaths(
                cycle,
                previous_cycle,
                case_fatality_rate,
                case_fatality_rate_hospitals_overwhelmed,
            )
            # Recalculate the estimated chance of infection
            cycle['est_actual_chance_of_infection'] = self.calculate_estimated_actual_chance_of_infection(
                cycle,
                previous_cycle,
                hospitalization_rate,
                model_parameters['population']
            )
            # Note the actual number of reported cases
            cycle['actual_reported'] = self.calculate_actual_reported(cycle, previous_cycle)
            # Recalculate how many people are susceptible at the end of the cycle
            cycle['ending_susceptible'] = self.calculate_ending_susceptible(cycle, previous_cycle,
                                                                               model_parameters['population'])
            # Recalculate how many hospital beds are left
            cycle['available_hospital_beds'] = self.calculate_available_hospital_beds(
                cycle,
                previous_cycle,
                max_hospital_capacity_factor,
                original_available_hospital_beds,
                hospital_capacity_change_daily_rate
            )
            # Prepare for the next iteration
            cycle_series.append(cycle)
            previous_cycle = cycle
            # Advance the clock
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
