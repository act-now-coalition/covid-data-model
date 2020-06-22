import numpy as np
from datetime import datetime, timedelta, date
from epiweeks import Week, Year
from enum import Enum

TEAM = 'CovidActNow'
MODEL = 'SEIR_CAN'

# type of target measures
TARGETS = ['cum death', 'inc death', 'inc hosp']

# names of target measures that will be used to generate metadata
TARGETS_TO_NAMES = {'cum death': 'cumulative deaths',
                    'inc death': 'incident deaths',
                    'inc hosp': 'incident hospitalizations'}

# units of forecast target.
FORECAST_TIME_UNITS = ['day', 'wk']
# number of weeks ahead for forecast.
FORECAST_WEEKS_NUM = 4
# Default quantiles required by CDC.
QUANTILES = np.concatenate([[0.01, 0.025], np.arange(0.05, 1, 0.05), [0.975, 0.99]])
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
    CUM_HOSP = 'cum hosp'
    INC_HOSP = 'inc hosp'


class ForecastTimeUnit(Enum):
    DAY = 'day'
    WK = 'wk'


class ForecastUncertainty(Enum):
    DEFAULT = 'default'
    NAIVE = 'naive'
