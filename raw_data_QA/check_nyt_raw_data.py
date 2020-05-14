import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from datetime import datetime
import calendar
import argparse
import pdb
import os
import shutil

def aggregate_df(df, args):
  #get all county level data points and add to get state level data points
  aggregation_functions = {args.new_cases_name:'sum', args.new_deaths_name:'sum', args.date_name:'first'}
  #add up all entries with same date
  df_new = df.groupby(df[args.date_name]).aggregate(aggregation_functions)
  #Get the number of new cases by taking the difference of cumulative cases given by NYT
  df_new['new_cases'] = df_new[args.new_cases_name].diff().fillna(0)
  df_new['new_deaths'] = df_new[args.new_deaths_name].diff().fillna(0)
  return df_new

def get_state(df, state, args):
  df_new = df.loc[df[args.state_name] == state]
  return df_new

def get_rmse(df1, df2, var):
  rmse = np.sqrt(mean_squared_error(df1[[var]], df2[[var]]))
  return round(rmse,2)

def get_equal_len_df(df1, df2):
  df2_length = len(df2.index)
  truncated_df1 = df1.head(df2_length)
  return truncated_df1

def get_compare_metrics(df1, df2, var):
  diff = []
  z_scores = []
  for i in range(len(df1.index)):
    var1 = df1[var].iloc[i]
    var2 = df2[var].iloc[i] 
    larger_var = max(var1, var2)
    smaller_var = min(var1, var2)
    if smaller_var == 0 and larger_var == 0:
      p_diff = 0
      z_score = 0
    #elif smaller_var == 0 and larger_var != 0: #may want to deal with small variations differently
    else:
      p_diff = (larger_var - smaller_var)/larger_var
      err = np.sqrt(larger_var)
      z_score = (larger_var - smaller_var)/err #to be added
    z_scores.append(z_score)
    diff.append(p_diff)
    number_of_days_percent_threshold = sum(i > args.percent_threshold for i in diff)
    average_Z = round(Average(z_scores), 2)
    latest_Z = round(Average(z_scores[:-args.n_days_z_score_mean]), 2)
  return diff, z_scores, average_Z, latest_Z, number_of_days_percent_threshold

def Average(list):
  if len(list) > 0:
    return sum(list)/len(list)
  else:
    return 0

def compare_data(var, df1, df2, df3, df1_name, df2_name, df3_name, args, state):
  #truncated df2 and df3 to df1 (this assumes df1 is the shortest)
  truncated_df2 = get_equal_len_df(df2, df1)
  truncated_df3 = get_equal_len_df(df3, df1)

  #Get comparison metrics
  rmse2 = get_rmse(df1, truncated_df2, var)
  rmse3 = get_rmse(truncated_df2, truncated_df3, var)
  diff_df2, z_df2, avg_z2, latest_avg_z2, days_over_z2 = get_compare_metrics(df1, truncated_df2, var)
  diff_df3, z_df3, avg_z3, latest_avg_z3, days_over_z3 = get_compare_metrics(df1, truncated_df3, var)

  #determine if data is different enough to plot for further investigation
  if args.use_latest:
    abnormal = rmse2 > args.rmse_threshold and rmse3 > args.rmse_threshold
  else:
    abnormal = rmse2 > args.rmse_threshold

  if abnormal:
    fig, ax = plt.subplots(3,1, sharex = True)
    fig.subplots_adjust(hspace = 0)
    markersize1 = 8
    markersize2 = 10
    markersize3 = 4
    markerstyle1 = '.'
    markerstyle2 = '.'
    markerstyle3 = '.'
    color1 = 'blue'
    color2 = 'orange'
    color3 = 'purple'
    fig.suptitle(var)
    ax[0].plot(df1.index.values, df1[var], color = color1, label = df1_name + ' NYT, RMSE: ' + str(rmse2) + ' $Z_{avg}$: ' + str(avg_z2),  markersize = markersize1, marker = markerstyle1, alpha = 0.5)
    ax[0].plot(df3.index.values, df3[var], color = color3, label = df3_name + ' NYT: RMSE: ' + str(rmse3) + ' $Z_{avg}$: ' + str(avg_z3), markersize = markersize3, marker = markerstyle3, alpha = 0.5)
    ax[0].plot(df2.index.values, df2[var],  color = color2, label = df2_name + ' NYT', markersize = markersize2, marker = markerstyle2, alpha = 0.5)
    ax[0].set(ylabel = var)
    ax[0].grid(True)

    ax[1].plot(df1.index.values, diff_df2, color = color2, markersize = markersize2, marker = markerstyle2, alpha = 0.5)
    ax[1].plot(df1.index.values, diff_df3, color = color3, markersize = markersize3, marker = markerstyle3, alpha = 0.5)
    ax[1].set(ylabel = 'Percent Difference')
    ax[1].grid(True)

    ax[2].plot(df1.index.values, z_df2, color = color2, markersize = markersize2, marker = markerstyle2, alpha = 0.5)
    ax[2].plot(df1.index.values, z_df3, color = color3, markersize = markersize3, marker = markerstyle3, alpha = 0.5)
    ax[2].set(ylabel='Z Score')
    ax[2].grid(True)

    plt.xticks(rotation=30)
    ax[0].legend(loc = 'upper left')
    plt.savefig(args.output_dir + '/' + state + '_' + var + '_compare.pdf', bbox_inches = 'tight')
    plt.close('all')
    return avg_z2, latest_avg_z2, days_over_z2, rmse2, rmse3
  else:
    return 0, 0, 0, 0, 0

def compare_county_state_plot(var, df1, df2, df1_name, df2_name, args, state):
  rmse2 = get_rmse(df1, df2, var)
  plt.title(state)
  plt.xlabel(var)
  plt.ylabel(args.updated_date_name)
  plt.plot(df1[args.updated_date_name], df1[var], color = 'blue', label = df1_name + ' NYT, RMSE: ' + str(rmse2), markersize = 8, marker = '.', alpha = 0.5)
  plt.plot(df2[args.updated_date_name], df2[var],  color = 'orange', label = df2_name + ' NYT', markersize = 8, marker = '.', alpha = 0.5)
  plt.xticks(rotation=30)
  plt.legend(loc = 'upper left')
  plt.grid(True)
  plt.savefig(state + '_raw_county_state' +  var + '_compare.png', bbox_inches='tight')
  plt.close('all')

def get_current_day():
  current_month = calendar.month_name[datetime.now().month]
  current_day = datetime.now().day
  return current_month + ' ' + str(current_day)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description = 'arg parser for process.py')
  parser.add_argument('-output_dir', '--output_dir', type = str, dest = 'output_dir', default = 'output', help = 'output_dir')
  parser.add_argument('-states', '--states', nargs = '+',  dest = 'states', default = ['All'], help = 'name of state to process')
  parser.add_argument('-state_name', '--state_name', type = str, dest = 'state_name', default = 'state', help = 'name of state variable in input csv')
  parser.add_argument('-new_cases_name', '--new_cases_name', type = str, dest = 'new_cases_name', default = 'cases', help = 'name of new cases column in input csv')
  parser.add_argument('-new_deaths_name', '--new_deaths_name', type = str, dest = 'new_deaths_name', default = 'deaths', help = 'name of new deaths name in input csv')
  parser.add_argument('-date_name', '--date_name', type = str, dest = 'date_name', default = 'date', help = 'name of datesin input csv')
  parser.add_argument('-updated_date_name', '--updated_date_name', type = str, dest = 'updated_date_name', default = 'Date', help = 'name of date variable that is computed in this script, really just renaming')
  parser.add_argument('-rmse_threshold', '--rmse_threshold', type = float, dest = 'rmse_threshold', default = 1, help = 'rmse threshold to trigger on')
  parser.add_argument('-percent_threshold', '--percent_threshold', type = float, dest = 'percent_threshold', default = .1, help = 'percent difference threshold to trigger on')
  parser.add_argument('-z_threshold', '--z_threshold', type = float, dest = 'z_threshold', default = 2, help = 'Z score threshold to trigger on')
  parser.add_argument('-n_days_over_threshold', '--n_days_over_threshold', type = int, default = 5, help = 'number of days to be over threshold to trigger on')
  parser.add_argument('-use_latest', '--use_latest', type = bool, default = False, help = 'set to true to use latest data * and * new data to determine unexpected changes')
  parser.add_argument('-n_days_z_score_mean', '--n_days_z_score_mean', type = int, default = 14, help = 'how many days in the past to average the z score to look for unexpcted changes to the data')
  args = parser.parse_args()

  #remove previous output directory if it exists
  if os.path.exists(args.output_dir):
    shutil.rmtree(args.output_dir)
  os.makedirs(args.output_dir)

  OLD_CAN_DATA = "data/us-counties-old.csv" #This will be used as the reference dataset which is compared to NEW_CAN_DATA and LATEST_NYT_DATA datasets
  NEW_CAN_DATA = "data/us-counties-new.csv"
  LATEST_NYT_DATA = "data/us-counties-latest.csv"
  LATEST_NYT_STATE_DATA = "data/us-states-latest.csv"
  variables = ['cases', 'deaths']

  #Get all states in input dataset if user asks for all states
  if 'All' in args.states:
    #could add start and end date here Natasha
    df1 = pd.read_csv(OLD_CAN_DATA)
    args.states = df1['state'].unique()
  
  #Iterate thru states
  for var in variables:
    z_avg_list, z_latest_avg_list, days_over_thres_list, states_list, rmse_new_list, rmse_latest_list = [], [], [], [], [], []
    for state in args.states:
      #Grab Data To Compare from CAN Data Caches
      df1 = get_state(pd.read_csv(OLD_CAN_DATA, parse_dates=[args.date_name]), state, args)
      df2 = get_state(pd.read_csv(NEW_CAN_DATA, parse_dates=[args.date_name]), state, args)
      df1_name = 'CAN DF1'
      df2_name = 'CAN DF2'

      #Grab Latest Data directly from NYT
      df3 = get_state(pd.read_csv(LATEST_NYT_DATA, parse_dates=[args.date_name]), state, args)
      df3_state = get_state(pd.read_csv(LATEST_NYT_STATE_DATA, parse_dates=[args.date_name]), state, args)
      df3_name = get_current_day()

      #Aggregate Datasets (i.e. combine counties to state level and calculate new cases and deaths)
      df1_ag = aggregate_df(df1, args)
      df2_ag = aggregate_df(df2, args)
      df3_ag = aggregate_df(df3, args)
      df3_state_ag = aggregate_df(df3_state, args)

      avg_z, latest_avg_z, days_over_z, rmse_new, rmse_latest = compare_data(var, df1_ag, df2_ag, df3_ag, df1_name, df2_name, df3_name, args, state)

      if avg_z > 0:
        z_avg_list.append(avg_z)
        z_latest_avg_list.append(latest_avg_z)
        days_over_thres_list.append(days_over_z)
        states_list.append(state)
        rmse_new_list.append(rmse_new)
        rmse_latest_list.append(rmse_latest)

    #Create meta-compare charts for all abnormal areas
    x_values = np.arange(len(states_list))
    fig, ax = plt.subplots()
    ax.bar(x_values, z_avg_list)
    ax.set_ylabel('Z Average')
    ax.set_xticklabels(states_list)
    plt.savefig(args.output_dir + '/meta_compare_' + var + '.pdf')
    plt.close('all')


    #compare_county_state_plot('new_cases', df_state_ag, df_county_ag, 'County Sum', 'State', args, state)


