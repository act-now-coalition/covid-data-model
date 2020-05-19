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

def load_Rt(fips):
    Rt = load_Rt_result(fips)
    Rt = Rt['Rt_MAP_composite']
    return Rt

def load_projection(fips):
    """

    """
    demo_output = pickle.load(open('demo_mapper_output.pkl', 'rb'))

    projection = {}
    projection.update(demo_output['compartments'])
    for measure in demo_output:
        if measure != 'compartments':
            for unit in demo_output[measure]:
                projection[measure + '__' + unit] = demo_output[measure][unit]

    return projection

