import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from datetime import datetime
import calendar
import argparse

def aggregate_df(df, args):
  df[args.new_cases_name] = pd.to_numeric(df[args.new_cases_name])
  df[args.new_deaths_name] = pd.to_numeric(df[args.new_deaths_name])
  aggregation_functions = {args.new_cases_name:'sum', args.new_deaths_name:'sum', args.updated_date_name:'first'}
  df_new = df.groupby(df[args.date_name]).aggregate(aggregation_functions)
  df_new['new_cases'] = df_new[args.new_cases_name].diff().fillna(0)
  df_new['new_deaths'] = df_new[args.new_deaths_name].diff().fillna(0)
  return df_new

def get_state(df, state, args):
  df_new = df.loc[df[args.state_name] == state]
  df_new[args.updated_date_name] = df[[args.date_name]]
  return df_new

def get_rmse(df1, df2, var):
  df1_length = len(df1.index)
  truncated_df2 = df2.head(df1_length)
  rmse = np.sqrt(mean_squared_error(df1[[var]], truncated_df2[[var]]))
  return round(rmse,2)

def make_plot(var, df1, df2, df3, df1_name, df2_name, df3_name, args, state):
  rmse2 = get_rmse(df1, df2, var)
  rmse3 = get_rmse(df2, df2, var)
  plt.title(state)
  plt.xlabel(var)
  plt.ylabel(args.updated_date_name)
  plt.plot(df1[args.updated_date_name], df1[var], color = 'blue', label = df1_name + ' NYT, RMSE: ' + str(rmse2), markersize = 8, marker = '.', alpha = 0.5)
  plt.plot(df2[args.updated_date_name], df2[var],  color = 'orange', label = df2_name + ' NYT, RMSE: ' + str(rmse3), markersize = 8, marker = '.', alpha = 0.5)
  #plt.plot(df3[args.updated_date_name], df3[var], alpha = 0.5, color = 'green', label = df3_name + ' NYT: RMSE: ' + str(rmse3), markersize = 10, marker = '.')
  plt.xticks(rotation=30)
  plt.legend(loc = 'upper left')
  plt.grid(True)
  plt.savefig(state + '_raw_' +  var + '_compare.pdf', bbox_inches='tight')
  plt.close('all')

def get_current_day():
  current_month = calendar.month_name[datetime.now().month]
  current_day = datetime.now().day
  return current_month + ' ' + str(current_day)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description = 'arg parser for process.py')
  #Output Dir------
  parser.add_argument('-output_dir', '--output_dir', type = str, dest = 'output_dir', default = 'output', help = 'output_dir')
  parser.add_argument('-state', '--state', nargs = '+',  dest = 'state', default = ['Idaho', 'Vermont', 'Texas'], help = 'name of state to process')
  parser.add_argument('-state_name', '--state_name', type = str, dest = 'state_name', default = 'state', help = 'name of state variable in input csv')
  parser.add_argument('-new_cases_name', '--new_cases_name', type = str, dest = 'new_cases_name', default = 'cases', help = 'name of new cases column in input csv')
  parser.add_argument('-new_deaths_name', '--new_deaths_name', type = str, dest = 'new_deaths_name', default = 'deaths', help = 'name of new deaths name in input csv')
  parser.add_argument('-date_name', '--date_name', type = str, dest = 'date_name', default = 'date', help = 'name of datesin input csv')
  parser.add_argument('-updated_date_name', '--updated_date_name', type = str, dest = 'updated_date_name', default = 'Date', help = 'name of date variable that is computed in this script, really just renaming')
  args = parser.parse_args()

  for state in args.state:
    df1 = get_state(pd.read_csv("data/us-counties-may8build.csv", parse_dates=[args.date_name]), state, args)
    df2 = get_state(pd.read_csv("data/us-counties-may11.csv", parse_dates=[args.date_name]), state, args)
    df3 = get_state(pd.read_csv("data/us-counties-latest.csv", parse_dates=[args.date_name]), state, args)
    df1_name = 'May 8 build'
    df2_name = 'May 11'
    df3_name = get_current_day()

    #pd.set_option('display.max_rows', None)

    df1_ag = aggregate_df(df1, args)
    df2_ag = aggregate_df(df2, args)
    df3_ag = aggregate_df(df3, args)
    make_plot('new_cases', df1_ag, df2_ag, df3_ag, df1_name, df2_name, df3_name, args, state)
    make_plot('new_deaths', df1_ag, df2_ag, df3_ag, df1_name, df2_name, df3_name, args, state)


