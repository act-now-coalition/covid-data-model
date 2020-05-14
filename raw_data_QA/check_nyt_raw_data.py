import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from datetime import datetime
import calendar
import argparse
import pdb
import os
import subprocess

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
  #df_new[args.updated_date_name] = df[[args.date_name]]
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
    #elif smaller_var == 0 and larger_var != 0:
    #  p_diff = (larger_var - smaller_var)/ smaller_var
    #  z_score = (larger_var - smaller_var) / smal
    else:
      p_diff = (larger_var - smaller_var)/larger_var
      err = np.sqrt(larger_var)
      z_score = (larger_var - smaller_var)/err #to be added
    z_scores.append(z_score)
    diff.append(p_diff)
  return diff, z_scores

def make_plot(var, df1, df2, df3, df1_name, df2_name, df3_name, args, state):
  #truncated df2 and df3 to df1 (this assumes df1 is the shortest)
  truncated_df2 = get_equal_len_df(df2, df1)
  truncated_df3 = get_equal_len_df(df3, df1)

  #This uses df2 as the reference!!!!!!!
  rmse2 = get_rmse(df1, truncated_df2, var)
  rmse3 = get_rmse(truncated_df2, truncated_df3, var)
  if rmse2 > 0.1 or rmse3 > 0.1:
    diff_df2, z_df2 = get_compare_metrics(df1, truncated_df2, var)
    diff_df3, z_df3 = get_compare_metrics(df1, truncated_df3, var)
    '''
    #plt.plot(df1.index.values, diff_df2, color = 'green')
    #plt.plot(df1.index.values, diff_df3, color = 'yellow')
    plt.plot(df1.index.values, z_df2, color = 'green')
    plt.plot(df1.index.values, z_df3, color = 'yellow')
    plt.xlabel('Date')
    plt.ylabel('Percent Difference')
    plt.savefig(args.output_dir + '/' + state + '_percentdiff.pdf')
    plt.close('all')
    plt.title(state)
    plt.xlabel(args.updated_date_name)
    plt.ylabel(var)
    print('HERE')
    plt.plot(df1.index.values, df1[var], color = 'orange', label = df1_name + ' NYT, RMSE: ' + str(rmse2), markersize = 15, marker = '.', alpha = 0.5)
    plt.plot(df2.index.values, df2[var],  color = 'blue', label = df2_name + ' NYT, RMSE: ' + str(rmse3), markersize = 8, marker = '*', alpha = 0.5)
    plt.plot(df3.index.values, df3[var], color = 'purple', label = df3_name + ' NYT: RMSE: ' + str(rmse3), markersize = 8, marker = 'd', alpha = 0.5)
    
    plt.xticks(rotation=30)
    plt.legend(loc = 'upper left')
    plt.grid(True)
    plt.savefig(args.output_dir + '/' + state + '_raw_' +  var + '_compare.png', bbox_inches='tight')
    plt.close('all')
    '''

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
    ax[0].plot(df1.index.values, df1[var], color = color1, label = df1_name + ' NYT, RMSE: ' + str(rmse2), markersize = markersize1, marker = markerstyle1, alpha = 0.5)
    ax[0].plot(df3.index.values, df3[var], color = color3, label = df3_name + ' NYT: RMSE: ' + str(rmse3), markersize = markersize3, marker = markerstyle3, alpha = 0.5)
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
    #plt.grid(True)
    plt.savefig(args.output_dir + '/' + state + '_compare.pdf', bbox_inches = 'tight')
    plt.close('all')

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
  args = parser.parse_args()

  #remove previous output directory if it exists
  if os.path.isdir(args.output_dir):
    subprocess.Popen("rm -r ./" + args.output_dir, stdout=subprocess.PIPE)
  os.mkdir(args.output_dir)
  if 'All' in args.states:
    #could add start and end date here Natasha
    df1 = pd.read_csv("data/us-counties-old.csv")
    args.states = df1['state'].unique()
    print(df1)
  
  for state in args.states:
    #Grab Data To Compare
    df1 = get_state(pd.read_csv("data/us-counties-old.csv", parse_dates=[args.date_name]), state, args)
    df2 = get_state(pd.read_csv("data/us-counties-new.csv", parse_dates=[args.date_name]), state, args)
    df1_name = 'CAN DF1'
    df2_name = 'CAN DF2'

    #Grab Latest Data
    df3 = get_state(pd.read_csv("data/us-counties-latest.csv", parse_dates=[args.date_name]), state, args)
    df3_state = get_state(pd.read_csv("data/us-states-latest.csv", parse_dates=[args.date_name]), state, args)
    df3_name = get_current_day()

    #Aggregate Datasets (i.e. combine counties to state level and calculate new cases and deaths to prepare for plotting)
    df1_ag = aggregate_df(df1, args)
    df2_ag = aggregate_df(df2, args)
    df3_ag = aggregate_df(df3, args)
    df3_state_ag = aggregate_df(df3_state, args)


    make_plot('cases', df1_ag, df2_ag, df3_ag, df1_name, df2_name, df3_name, args, state)
    make_plot('deaths', df1_ag, df2_ag, df3_ag, df1_name, df2_name, df3_name, args, state)
    #compare_county_state_plot('new_cases', df_state_ag, df_county_ag, 'County Sum', 'State', args, state)
    #compare_county_state_plot('new_deaths', df_state_ag, df_county_ag, 'County Sum', 'State', args, state)


