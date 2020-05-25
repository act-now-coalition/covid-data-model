import pandas as pd
import numpy as np
import enum as Enum
import dill as pickle
from pyseir.inference.fit_results import load_Rt_result


def load_population_size(fips):
    """
    Load population size for a community.
    """
    return 1000

def load_rt(fips):
    Rt = load_Rt_result(fips)
    Rt = Rt['Rt_MAP_composite']
    Rt = Rt.dropna()
    return Rt

def load_projection(fips):
    """

    """
    demo_output = pickle.load(open('demo_mapper_output.pkl', 'rb'))

    projection = list()
    projection.append(demo_output['compartments'])

    for measure in demo_output:
        if measure != 'compartments':
            for unit in demo_output[measure]:
                s = demo_output[measure][unit]
                projection.append(s.rename(measure + '__' + unit))

    projection = pd.concat(projection, axis=1)

    return projection


def load_pcr_stats():
    """
    load pcr sensitivity and specificity
    """
    return


def load_antibody_stats():
    """
    load antibody test sensitivity and specificity
    """

    return

