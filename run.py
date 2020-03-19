import csv
import logging
import numpy as np
import pandas as pd
import pprint
import datetime

# Fixed Data Sources
beds = pd.read_csv("data/beds.csv")
populations = pd.read_csv("data/populations.csv")

# Modeling Assumptions
r0_initial = 2.8
hospitalization_rate = .05
case_fatality_rate = .015
case_fatality_rate_hospitals_overwhelmed = .015
hospital_capacity_change_daily_rate = 1.05
initial_hospital_bed_utilization = .5

model_interval = 4
rolling_intervals_for_current_infected = 3

logging.basicConfig(level=logging.DEBUG)

def get_population(province_state, country_region):
    matching_pops = populations[(populations["Province/State"] == province_state) & (populations["Country/Region"] == country_region)]
    return int(matching_pops.iloc[0].at["Population"])

def get_beds(province_state, country_region):
    matching_beds = beds[(beds["Province/State"] == province_state) & (beds["Country/Region"] == country_region)]
    beds_per_mille = matching_beds.iloc[0].at["Beds Per 1000"]
    return int(beds_per_mille * get_population(province_state, country_region) / 1000)

def get_snapshot(date, province_state, country_region):
    snapshot_filename = 'data/{}.csv'.format(date.strftime('%m-%d-%Y'))
    logging.debug('Loading: {}'.format(snapshot_filename))
    full_snapshot = pd.read_csv(snapshot_filename)
    filtered_snapshot = full_snapshot[(full_snapshot["Province/State"] == province_state) & (full_snapshot["Country/Region"] == country_region)]
    pprint.pprint(filtered_snapshot)

    confirmed = 0
    deaths = 0
    recovered = 0

    try:
        row = filtered_snapshot.iloc[0]
        confirmed = row.at['Confirmed']
        deaths = row.at['Deaths']
        recovered = row.at['Recovered']
    except IndexError as e:
        pass

    return {'confirmed': confirmed, 'deaths': deaths, 'recovered': recovered}

def forecast_region(province_state, country_region, iterations):
    logging.info('Building results for {} in {}'.format(province_state, country_region))
    pop = get_population(province_state, country_region)
    beds = get_beds(province_state, country_region)
    logging.debug('This location has {} beds for {} people'.format(beds, pop))

    logging.debug('Loading daily report from {} days ago'.format(model_interval))


    cols = ['Note',
            'Date',
            'Effective R0',
            'Beginning Susceptible',
            'New Infected',
            'Previously Infections',
            'Recovered or Died',
            'Ending Susceptible',
            'Actual Reported (Hospitalized?)',
            'Predicted Hospitalized',
            'Cumulative Infected',
            'Cumulative Deaths',
            'Available Hospital Beds',
            'S&P 500',
            'Estimated Actual Chance of Infection',
            'Predicted Chance of Infection',
            'Cumulative Predicted Chance of Infection',
            'R0',
            '% Susceptible']
    rows = []


    effective_r0 = r0_initial
    previous_confirmed = 0
    previous_ending_susceptible = pop
    previous_newly_infected = 0
    current_infected_series = []
    recovered_or_died = 0
    cumulative_infected = 0
    cumulative_deaths = 0
    available_hospital_beds = beds * (1 - initial_hospital_bed_utilization)

    # @TODO: See if today's data is already available. If so, don't subtract an additional day.
    today = datetime.date.today() - datetime.timedelta(days = 1)

    snapshot_date = today - datetime.timedelta(days = model_interval * rolling_intervals_for_current_infected)

    # Step through existing empirical data
    while snapshot_date <= today:
        snapshot = get_snapshot(snapshot_date, province_state, country_region)

        # Use an empirical R0, if available. Otherwise, use the default.
        if previous_confirmed > 0:
            effective_r0 = snapshot['confirmed'] / previous_confirmed
        previous_confirmed = snapshot['confirmed']

        if previous_newly_infected > 0:
            # If we have previously known cases, use the R0 to estimate newly infected cases.
            newly_infected = previous_newly_infected * effective_r0 * previous_ending_susceptible / pop
        else:
            # We assume the first positive cases were exclusively hospitalized ones.
            actual_infected_vs_tested_positive = 1 / hospitalization_rate
            newly_infected = snapshot['confirmed'] * actual_infected_vs_tested_positive

        # Assume infected cases from before the rolling interval have concluded.
        if (len(current_infected_series) >= 4):
            recovered_or_died = recovered_or_died + current_infected_series[-rolling_intervals_for_current_infected-1]

        previously_infected = sum(current_infected_series[-rolling_intervals_for_current_infected:])
        cumulative_infected += newly_infected
        predicted_hospitalized = newly_infected * hospitalization_rate

        if (available_hospital_beds > predicted_hospitalized):
            cumulative_deaths += newly_infected * case_fatality_rate
        else:
            cumulative_deaths += newly_infected * case_fatality_rate_hospitals_overwhelmed

        est_actual_chance_of_infection = (snapshot['confirmed'] / hospitalization_rate * 2) / pop
        predicted_chance_of_infection = None

        row = ('',
               snapshot_date,
               effective_r0,
               previous_ending_susceptible,  # Beginning susceptible
               newly_infected,
               previously_infected,
               recovered_or_died,
               pop - newly_infected - previously_infected - recovered_or_died,  # Ending Susceptible
               snapshot['confirmed'],
               predicted_hospitalized,
               cumulative_infected,
               cumulative_deaths,
               available_hospital_beds,
               None,  # S&P 500
               snapshot['confirmed'],
               est_actual_chance_of_infection,
               predicted_chance_of_infection,
               None,
               None,
               None)

        current_infected_series.append(newly_infected)

        available_hospital_beds *= hospital_capacity_change_daily_rate
        snapshot_date += datetime.timedelta(days=model_interval)

    for i in range(0, iterations):
        row = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S')
        rows.append(row)
        snapshot_date += datetime.timedelta(days=model_interval)

    forecast = pd.DataFrame(rows, columns=cols)

    pprint.pprint(forecast)



#forecast_region('New South Wales', 'Australia', 50)
#forecast_region('Queensland', 'Australia', 50)
forecast_region('California', 'US', 50)
