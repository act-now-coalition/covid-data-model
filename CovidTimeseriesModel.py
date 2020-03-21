import logging
import pandas as pd
import datetime
class CovidTimeseriesModel:

	# Initializer / Instance Attributes
	def __init__(self):
		logging.basicConfig(level=logging.CRITICAL)


	def get_population(self, state, country, populations):
			matching_pops = populations[(populations["state"] == state) & (
		    populations["country"] == country)]
			return int(matching_pops.iloc[0].at["population"])


	def get_beds(self, state, country, beds_data, populations_data):
		matching_beds = beds_data[(beds_data["state"] == state) &
		                          (beds_data["country"] == country)]
		beds_per_mille = float(matching_beds.iloc[0].at["bedspermille"])
		return int(round(beds_per_mille * self.get_population(state, country, populations_data) / 1000))


	def get_snapshot(self, date, state, country, full_timeseries):
	# First, attempt to pull the state-level data without aggregating.
		filtered_timeseries = full_timeseries[(full_timeseries["state"] == state) & (
		    full_timeseries["country"] == country) & (full_timeseries['date'] == date.strftime('%Y-%m-%d')) & (full_timeseries["county"].isna())]

		#pprint.pprint(filtered_timeseries)

		# Switch to aggregating across counties if that returns no cases.
		if int(filtered_timeseries['cases'].sum()) == 0:
		      filtered_timeseries = full_timeseries[(full_timeseries["state"] == state) & (
		          full_timeseries["country"] == country) & (full_timeseries['date'] == date.strftime('%Y-%m-%d'))]

		#pprint.pprint(filtered_timeseries)

		confirmed = None
		deaths = 0
		recovered = 0

		try:
		    #row = filtered_snapshot.iloc[0]
		    confirmed = int(filtered_timeseries['cases'].sum())
		    deaths = int(filtered_timeseries['recovered'].sum())
		    recovered = int(filtered_timeseries['deaths'].sum())
		except IndexError as e:
		    pass

		return {'confirmed': confirmed, 'deaths': deaths, 'recovered': recovered}

	def calculate_r(self, snapshot, previous_snapshot, r0):
	    # Calculate the r0 value based on the current and past number of confirmed cases
	    if snapshot['confirmed'] is not None:
	        if previous_snapshot['confirmed'] > 0:
	            return snapshot['confirmed'] / previous_snapshot['confirmed']
	    return r0


	def calculate_effective_r(self, snapshot, previous_snapshot, pop):
	    # Calculate effective r by accounting for herd-immunity
	    return snapshot['r'] * (previous_snapshot['ending_susceptible'] / pop)


	def calculate_newly_infected(self, snapshot, previous_snapshot, pop, initial_hospitalization_rate):
	    if previous_snapshot['newly_infected'] > 0:
	        # If we have previously known cases, use the R0 to estimate newly infected cases.
	        newly_infected = previous_snapshot['newly_infected'] * self.calculate_effective_r(snapshot, previous_snapshot, pop)
	    else:
	        # TODO: Review. I'm having trouble following this block
	        # We assume the first positive cases were exclusively hospitalized ones.
	        actual_infected_vs_tested_positive = 1 / initial_hospitalization_rate  # ~20
	        if snapshot['confirmed'] is not None:
	            confirmed = snapshot['confirmed']
	        newly_infected = snapshot['confirmed'] *             actual_infected_vs_tested_positive
	    return newly_infected


	def calculate_currently_infected(self, snapshot_series, rolling_intervals_for_current_infected):
	    # Calculate the number of people who have been infected but are no longer infected (one way or another)
	    # Better way:
	    return sum(
	            ss['newly_infected'] for ss in snapshot_series[-rolling_intervals_for_current_infected:]
	    )


	def hospital_is_overloaded(self, snapshot, previous_snapshot):
	    # Determine if a hospital is overloaded. Trivial now, but will likely become more complicated in the near future
	    return previous_snapshot['available_hospital_beds'] > snapshot['predicted_hospitalized']


	def calculate_cumulative_deaths(self, snapshot, previous_snapshot, case_fatality_rate, case_fatality_rate_hospitals_overwhelmed, logged_overwhelmed, state):
	    # If the number of hospital beds available is exceeded by the number of patients that need hospitalization,
	    #  the death rate increases
	    # Can be significantly improved in the future
	    if self.hospital_is_overloaded(snapshot, previous_snapshot):
	        return previous_snapshot['cumulative_deaths'] + round(snapshot['newly_infected'] * case_fatality_rate)
	    else:
	        return previous_snapshot['cumulative_deaths'] +             round(
	                snapshot['newly_infected'] * (case_fatality_rate + case_fatality_rate_hospitals_overwhelmed)
	            )


	def calcluate_recovered_or_died(self, snapsnot, previous_snapshot, snapshot_series, rolling_intervals_for_current_infected):
	    # Recovered or died (RoD) is a cumulative number. We take the number of RoD from last cycle, and add to it
	    #  the number of individuals who were newly infected a set number of cycles ago (rolling_intervals_for_current_infected)
	    #  (RICI). It is assumed that after the RICI, an infected interval has either recovered or died.
	    # Can be improved in the future
	    if len(snapshot_series) >= (rolling_intervals_for_current_infected + 1):
	        return previous_snapshot['recovered_or_died'] +             snapshot_series[-(rolling_intervals_for_current_infected + 1)]['newly_infected']
	    else:
	        return previous_snapshot['recovered_or_died']


	def calculate_ending_susceptible(self, snapshot, previous_snapshot, pop):
	    # Number of people who haven't been sick yet
	    return round(
	        pop - (snapshot['newly_infected'] + snapshot['currently_infected'] + snapshot['recovered_or_died'])
	    )


	def calculate_estimated_actual_chance_of_infection(self, snapshot, previous_snapshot, hospitalization_rate, pop):
	    # Reflects the model logic, but probably needs to be changed
	    if snapshot['confirmed'] is not None:
	        return ((snapshot['confirmed'] / hospitalization_rate) * 2) / pop
	    return None


	def calculate_actual_reported(self, snapshot, previous_snapshot):
	    # Needs to account for creating synthetic data for missing records
	    if snapshot['confirmed'] is not None:
	        return round(snapshot['confirmed'])
	    return 0


	def calculate_available_hospital_beds(self, snapshot, previous_snapshot, max_hospital_capacity_factor, original_available_hospital_beds, hospital_capacity_change_daily_rate):
	    available_hospital_beds = previous_snapshot['available_hospital_beds']
	    if available_hospital_beds < max_hospital_capacity_factor * original_available_hospital_beds:
	            # Hospitals can try to increase their capacity for the sick
	            available_hospital_beds *= hospital_capacity_change_daily_rate
	    return available_hospital_beds

	def iterate_model(self, iterations, country, state, pop, beds, model_parameters):
	    """
	    The guts. Creates the initial conditions, then iterates the model over the data for a specified number
	    of iterations
	    """
	    # Unpack the model parameters
	    r0 = model_parameters['r0']
	    model_interval = model_parameters['model_interval']
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
	    # Set the initial snapshot date
	    init_date = today - datetime.timedelta(days=model_interval * rolling_intervals_for_current_infected)
	    original_available_hospital_beds = round(beds * (1 - initial_hospital_bed_utilization), 0)

	    init_snapshot = {
	        'date': init_date,
	        'r': r0,
	        'confirmed': 0,
	        'actual_reported': 0,
	        'current_infected': 0,
	        'est_actual_chance_of_infection': 0,
	        'newly_infected': 0,
	        'currently_infected': 0,
	        'cumulative_infected': 0,
	        'cumulative_deaths': 0,
	        'recovered_or_died': 0,
	        'ending_susceptible': pop,
	        'predicted_hospitalized': 0,
	        'available_hospital_beds': original_available_hospital_beds

	    }
	    snapshot_series = [init_snapshot, ]
	    previous_snapshot = init_snapshot
	    snapshot_date = init_date
	    logged_overwhelmed = False # TODO: Tristan broke this a little. Python 'passes by value', so this doesn't get updated in the subroutine
	    while True:
	        # TODO: If we build the list of snapshots ahead of time, we can change the while loop to a for loop. Dunno if we need to do that
	        # Step through existing empirical data
	        if snapshot_date <= today:
	            snapshot = self.get_snapshot(snapshot_date, state, country, model_parameters['full_timeseries'])
	        # If the snapshot date is past today's date, we're projecting from here on out
	        else:
	            snapshot = {'confirmed': None, 'deaths': None, 'recovered': None}
	        snapshot['date'] = snapshot_date

	        # Run the model until enough iterations are complete.
	        if snapshot['date'] > today + datetime.timedelta(days=iterations * model_interval):
	            break  # @TODO change to work predictively

	        logging.debug('Calculating values for {}'.format(snapshot['date']))

	        # Calculate the r0 value
	        snapshot['r'] = self.calculate_r(snapshot, previous_snapshot, r0)
	        # Calculate the number of newly infected cases
	        snapshot['newly_infected'] = self.calculate_newly_infected(
	            snapshot,
	            previous_snapshot,
	            pop,
	            initial_hospitalization_rate
	        )
	        # Assume infected cases from before the rolling interval have concluded.
	        snapshot['recovered_or_died'] = self.calcluate_recovered_or_died(
	            snapshot,
	            previous_snapshot,
	            snapshot_series,
	            rolling_intervals_for_current_infected
	        )
	        # Calculate the number of people who have already been infected
	        snapshot['currently_infected'] = self.calculate_currently_infected(
	            snapshot_series,
	            rolling_intervals_for_current_infected
	        )
	        # Calculate the cumulative number of infected individuals
	        snapshot['cumulative_infected'] = previous_snapshot['cumulative_infected'] + snapshot['newly_infected']
	        # Predict the number of patients that will require hospitilzation
	        snapshot['predicted_hospitalized'] = snapshot['newly_infected'] * hospitalization_rate
	        # Calculate the number of cumulative deaths
	        snapshot['cumulative_deaths'] = self.calculate_cumulative_deaths(
	            snapshot,
	            previous_snapshot,
	            case_fatality_rate,
	            case_fatality_rate_hospitals_overwhelmed,
	            logged_overwhelmed,
	            state
	        )
	        # Recalculate the estimated chance of infection
	        snapshot['est_actual_chance_of_infection'] = self.calculate_estimated_actual_chance_of_infection(
	            snapshot,
	            previous_snapshot,
	            hospitalization_rate,
	            pop
	        )
	        # Note the actual number of reported cases
	        snapshot['actual_reported'] = self.calculate_actual_reported(snapshot, previous_snapshot)
	        # Recalculate how many people are susceptible at the end of the cycle
	        snapshot['ending_susceptible'] = self.calculate_ending_susceptible(snapshot, previous_snapshot, pop)
	        # Recalculate how many hospital beds are left
	        snapshot['available_hospital_beds'] = self.calculate_available_hospital_beds(
	            snapshot,
	            previous_snapshot,
	            max_hospital_capacity_factor,
	            original_available_hospital_beds,
	            hospital_capacity_change_daily_rate
	        )
	        # Prepare for the next iteration
	        snapshot_series.append(snapshot)
	        previous_snapshot = snapshot
	        # Advance the clock
	        snapshot_date += datetime.timedelta(days=model_interval)
	    return snapshot_series

	def forecast_region(self, state, country, iterations, model_parameters):
	    logging.info('Building results for {} in {}'.format(state, country))
	    pop = self.get_population(state, country,  model_parameters['populations_data'])
	    beds = self.get_beds(state, country, model_parameters['beds_data'], model_parameters['populations_data'])
	    logging.debug('This location has {} beds for {} people'.format(beds, pop))
	    logging.debug('Loading daily report from {} days ago'.format(model_parameters['model_interval']))

	    snapshot_series = self.iterate_model(
	        iterations,
	        country,
	        state,
	        pop,
	        beds,
	        model_parameters
	    )

	    forecast = pd.DataFrame({
	        'Note': ['' for s in snapshot_series],
	        'Date': [s['date'] for s in snapshot_series],
	        'Timestamp': [datetime.datetime(year=s['date'].year, month=s['date'].month, day=s['date'].day).timestamp() for s in snapshot_series],
	        'Eff. R0': [s['r'] for s in snapshot_series],
	        'Beg. Susceptible': [s['ending_susceptible'] for s in snapshot_series],
	        'New Inf.': [s['newly_infected'] for s in snapshot_series],
	        'Curr. Inf.': [s['currently_infected'] for s in snapshot_series],
	        'Recov. or Died': [s['recovered_or_died'] for s in snapshot_series],
	        'End Susceptible': [s['ending_susceptible'] for s in snapshot_series],
	        'Actual Reported': [s['actual_reported'] for s in snapshot_series],
	        'Pred. Hosp.': [s['predicted_hospitalized'] for s in snapshot_series],
	        'Cum. Inf.': [s['cumulative_infected'] for s in snapshot_series],
	        'Cum. Deaths': [s['cumulative_deaths'] for s in snapshot_series],
	        'Avail. Hosp. Beds': [s['available_hospital_beds'] for s in snapshot_series],
	        'S&P 500': [None for s in snapshot_series],
	        'Est. Actual Chance of Inf.': [s['est_actual_chance_of_infection'] for s in snapshot_series],
	        'Pred. Chance of Inf.': [None for s in snapshot_series],
	        'Cum. Pred. Chance of Inf.': [None for s in snapshot_series],
	        'R0': [None for s in snapshot_series],
	        '% Susceptible': [None for s in snapshot_series]
	    })
	    return forecast
