import numpy as np
from datetime import datetime, timedelta, date
from epiweeks import Week, Year
from enum import Enum

TEAM = 'CovidActNow'
MODEL = 'SEIR_CAN'

# type of target measures
TARGETS_AND_UNITS = [('cum death', 'wk'),
                     ('inc death', 'wk'),
                     ('cum death', 'day'),
                     ('inc death', 'day'),
                     ('inc case', 'wk'),
                     ('inc case', 'day'),
                     ('inc hosp', 'day'),
                     ('inc hosp', 'wk')]

# type of target measures
TARGETS_AND_UNITS_REPORT = [('cum death', 'wk'),
                            ('inc death', 'wk'),
                            ('inc case', 'wk'),
                            ('inc hosp', 'wk')]

# names of target measures that will be used to generate metadata
TARGETS_TO_NAMES = {'cum death': 'cumulative deaths',
                    'inc death': 'incident deaths',
                    'inc case': 'incident confirmed cases',
                    'inc hosp': 'incident hospitalizations'}

# number of weeks ahead for forecast.
FORECAST_WEEKS_NUM = 6
# Default quantiles required by CDC.
QUANTILES_STATES = np.concatenate([[0.01, 0.025], np.arange(0.05, 1, 0.05), [0.975, 0.99]])
QUANTILES_COUNTIES = np.concatenate([[0.01, 0.025], np.arange(0.05, 1, 0.05), [0.975, 0.99]])
# Time of forecast, default date when this runs.
FORECAST_DATE = datetime.today() - timedelta(days=1)
# Next epi week. Epi weeks starts from Sunday and ends on Saturday.
#if forecast date is Sunday or Monday, next epi week is the week that starts
#with the latest Sunday.
if FORECAST_DATE.weekday() in (0, 6):
    NEXT_EPI_WEEK = Week(Year.thisyear().year, Week.thisweek().week)
else:
    NEXT_EPI_WEEK = Week(Year.thisyear().year, Week.thisweek().week + 1)
COLUMNS = ['forecast_date', 'location', 'target', 'type',
           'target_end_date', 'quantile', 'value']

DATE_FORMAT = '%Y-%m-%d'

class Target(Enum):
    CUM_DEATH = 'cum death'
    INC_DEATH = 'inc death'
    INC_CASE = 'inc case'
    INC_HOSP = 'inc hosp'

class ForecastTimeUnit(Enum):
    DAY = 'day'
    WK = 'wk'

class ForecastUncertainty(Enum):
    DEFAULT = 'default'
    NAIVE = 'naive'
