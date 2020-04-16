import copy
from dataclasses import dataclass
from typing import Dict, List, Callable, Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split

from pyseir.parameters.parameter_ensemble_generator import ParameterEnsembleGenerator
from pyseir.models import suppression_policies as sp
from pyseir.models import seir_model as sm


@dataclass
class ContinuousParameter:
    """
    This class is used by the OutcomesSampler to define a range for a model parameter
    over which it will sample points and evaluate the SEIRModel on. 

    Parameters
    ----------
    parameter_name: str
        Name of the model parameter
    lower_bound: float
        Lower bound of the range to sample the model parameter
    upper_bound: float
        Upper bound of the range to sample the model parameter
    alpha: float
        Scale parameter of a Beta distribution to sample the model parameter from
    beta: float
        Scale parameter of a Beta distribution to sample the model parameter from
    """

    parameter_name: str
    lower_bound: float
    upper_bound: float
    alpha: float = 1
    beta: float = 1

    def sample(self, num_samples: int, random_state: int=None) -> np.array:
        """
        Sample points between the lower_bound and the upper_bound using a Beta distribution. 
        The default values of alpha and beta correspond to a uniform distribution.

        Parameters
        ----------
        num_samples: int
            The number of samples to draw between lower_bound and upper_bound
        random_state: int
            Seed for numpy.random.RandomState
        Returns
        -------
        samples: numpy array
            An array with num_samples between lower_bound and upper_bound
        """
        scale = (self.upper_bound - self.lower_bound) 
        shift = self.lower_bound
        rs = np.random.RandomState(random_state)
        samples = rs.beta(a=self.alpha, b=self.beta, size=num_samples)
        return scale * samples + shift


class OutcomesSampler:
    """
    Implements a class for sampling points in parameter space to evaluate a SEIRModel at.

    Parameters
    ----------
    parameter_generator: ParameterEnsembleGenerator
        Generate parameters for SEIR modeling.
    parameter_space: List[ContinuousParameter]
        A collection of ContinuousParameters that defines the space the SEIRModel will be 
        evaluated on.
    outcome_fs: Dict[str, Callable]
        A dictionary of functions to evaluate on rollouts of the SEIRModel which return
        outcomes of those rollouts
    num_samples: int
        The number of points to sample in the parameter_space
    n_jobs: int
        Controls the number of processes joblib will use. Defaults to -1 which will use all CPU's
    """

    def __init__(self,
                 parameter_generator: ParameterEnsembleGenerator,    
                 parameter_space: List[ContinuousParameter],
                 outcome_fs: Dict[str, Callable],
                 num_samples: int=1000,
                 n_jobs=-1):

        self.num_samples = num_samples
        self.parameter_space = parameter_space
        self.n_jobs = n_jobs

        self.parameter_generator = parameter_generator
        self.parameter_defaults = self.parameter_generator.get_average_seir_parameters()

        self.outcomes_df = self._random_parameters()
        outcomes = self.get_outcomes(outcome_fs)
        self.outcomes_df = pd.merge(self.outcomes_df, outcomes, left_index=True, right_index=True)

    def _random_parameters(self) -> pd.DataFrame:
        """
        Draws random points in the parameter_space

        Returns
        -------
        samples: DataFrame
            A DataFrame with num_samples of points in parameter_space
        """

        samples = {i.parameter_name: i.sample(self.num_samples) for i in self.parameter_space}
        return pd.DataFrame(samples)

    def _evaluate_model(self, parameters: Dict[str, Any]) -> sm.SEIRModel:
        """
        Evaluates the SEIRModel at one point in parameter_space

        Parameters
        ----------
        parameters: dict
            SEIRModel parameters
        
        Returns
        -------
        model: SEIRModel
            A SEIRModel which has been run
        """

        model = sm.SEIRModel(**parameters)
        model.run()
        return model
    
    def _rollout(self,
                 x: np.array, 
                 parameters: dict,
                 outcome_fs: Dict[str, Callable]) -> pd.DataFrame:
        """
        Evaluates SEIRModel and calculates outcomes from the rollout of the model

        Parameters
        ----------
        x: numpy array
            A point in parameter_space
        parameters: dict
            A dictionary of SEIRModel parameters
        outcome_fs: Dict[str, Callable]
            A dictionary of functions that calculate outcomes for a given rollout of SEIRModel

        Returns
        -------
        outcomes: dict
            A dictionary of outcomes from a rollout of SEIRModel
        """

        fips = self.parameter_generator.fips
        t_list = self.parameter_generator.t_list

        for p, v in zip(self.parameter_space, x):
            if p.parameter_name == "suppression_policy":
                parameters['suppression_policy'] = \
                    sp.generate_empirical_distancing_policy(t_list, fips=fips, future_suppression=v)
            else:
                parameters[p.parameter_name] = v
        
        model = self._evaluate_model(parameters)
        outcomes = {k: f(model.results[k.split("-")[0]]) for k, f in outcome_fs.items()}
        return outcomes

    def get_outcomes(self, outcome_fs: Dict[str, Callable]) -> pd.DataFrame:
        """
        Evaluates SEIRModel at points in parameter_space and returns a DataFrame of outcomes

        Parameters
        ----------
        outcome_fs: Dict[str, Callable]
            A dictionary of functions that calculate outcomes for a given rollout of SEIRModel

        Returns
        -------
        outcomes_df: DataFrame
            A DataFrame with points in parameter space and outcomes calculated from a rollout of 
            SEIRModel at each point 
        """

        parameters = copy.deepcopy(self.parameter_defaults)
        space = self.outcomes_df[[p.parameter_name for p in self.parameter_space]].values
        outcomes = Parallel(n_jobs=self.n_jobs)(delayed(self._rollout)(x, parameters, outcome_fs) for x in space)
        return pd.DataFrame(outcomes)


class OutcomeModels:
    """
    Takes a DataFrame of SEIRModel parameters and outcomes and fits a function to it.
    The fitted model can be used to explore how outcomes respond to changes in SEIRModel parameters.

    Parameters
    ----------
    outcome_samples: DataFrame
        A DataFrame of SEIRModel outcomes at different points in parameter space
    parameter_names: List[str]
        List of the names of the parameters that define the space the SEIRModel
        was evaluated in
    fn_approximator: Callable
        A function that takes a training set and returns a fitted model
    n_jobs: int
        Controls the number of processes joblib will use. Defaults to -1 which will use all CPU's
    random_state: int
        Seed for numpy.random.RandomState
    """

    def __init__(self, 
                 outcome_samples: pd.DataFrame, 
                 parameter_names: List[str],
                 fn_approximator: Callable,
                 n_jobs=-1,
                 random_state: int=None):
        self.outcome_samples = outcome_samples
        self.parameter_names = parameter_names
        self.outcome_names = [i for i in self.outcome_samples.columns if i not in self.parameter_names]
        self.train_set, self.test_set = train_test_split(self.outcome_samples, test_size=0.1, random_state=random_state)
        outcome_models = \
            Parallel(n_jobs=-1)(delayed(fn_approximator)(oc, self.train_set[parameter_names], self.train_set[oc]) 
                                      for oc in self.outcome_names)
        self.outcome_models = dict(outcome_models)

    
        

        


