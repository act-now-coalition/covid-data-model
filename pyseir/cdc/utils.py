from enum import Enum

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
    NAIVE = 'native'

def target_column_name(num, target, time_unit):
    """

    """
    num = list(num)
    for n in num:
        yield f'{int(n)} {time_unit.value} ahead {target.value}'

