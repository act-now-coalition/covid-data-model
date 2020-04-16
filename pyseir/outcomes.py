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
    parameter_name: str
    lower_bound: float
    upper_bound: float
    alpha: float = 1
    beta: float = 1

    def sample(self, num_samples):
        scale = (self.upper_bound - self.lower_bound) 
        shift = self.lower_bound
        samples = np.random.beta(a=self.alpha, b=self.beta, size=num_samples)
        return scale * samples + shift


class OutcomesSampler:

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
        samples = {i.parameter_name: i.sample(self.num_samples) for i in self.parameter_space}
        return pd.DataFrame(samples)

    def _evaluate_model(self, parameters: Dict[str, Any]) -> sm.SEIRModel:
        model = sm.SEIRModel(**parameters)
        model.run()
        return model
    
    def _rollout(self,
                 x: np.array, 
                 parameters: dict,
                 outcome_f: Dict[str, Callable]) -> pd.DataFrame:

        fips = self.parameter_generator.fips
        t_list = self.parameter_generator.t_list

        for p, v in zip(self.parameter_space, x):
            if p.parameter_name == "suppression_policy":
                parameters['suppression_policy'] = \
                    sp.generate_empirical_distancing_policy(t_list, fips=fips, future_suppression=v)
            else:
                parameters[p.parameter_name] = v
        
        model = self._evaluate_model(parameters)
        outcomes = {k: f(model.results[k.split("-")[0]]) for k, f in outcome_f.items()}
        return outcomes

    def get_outcomes(self, outcome_fs: Dict[str, Callable]) -> pd.DataFrame:
        parameters = copy.deepcopy(self.parameter_defaults)
        space = self.outcomes_df[[p.parameter_name for p in self.parameter_space]].values
        outcomes = Parallel(n_jobs=self.n_jobs)(delayed(self._rollout)(x, parameters, outcome_fs) for x in space)
        return pd.DataFrame(outcomes)


class OutcomeModels:

    def __init__(self, 
                 outcome_samples: pd.DataFrame, 
                 parameter_space: List[ContinuousParameter],
                 fn_approximator: Callable,
                 n_jobs=-1):
        self.outcome_samples = outcome_samples
        self.parameter_space = parameter_space
        parameter_names = [p.parameter_name for p in self.parameter_space]
        outcome_names = [i for i in self.outcome_samples.columns if i not in parameter_names]
        self.train_set, self.test_set = train_test_split(self.outcome_samples, test_size=0.1)
        outcome_models = \
            Parallel(n_jobs=-1)(delayed(fn_approximator)(oc, self.train_set[parameter_names], self.train_set[oc]) 
                                      for oc in outcome_names)
        self.outcome_models = dict(outcome_models)

    
        

        


