import scipy
import numpy as np
import pandas as pd
from pyseir import load_data
from pyseir.parameters.parameter_ensemble_generator import ParameterEnsembleGenerator

hosp_rate_data = None


class ParameterEnsembleGeneratorAge(ParameterEnsembleGenerator):
    """
    Generate ensembles of parameters for SEIR modeling with age structure.

    Parameters
    ----------
    fips: str
        County or state fips code.
    N_samples: int
        Integer number of samples to generate.
    t_list: array-like
        Array of times to integrate against.
    I_initial: int
        Initial infected case count to consider.
    suppression_policy: callable(t): pyseir.model.suppression_policy
        Suppression policy to apply.
    contact_matrix_data: dict
        Contains contact matrix, age bin edges and corresponding age
        distribution of relevant fips. With format:
        {<fips>:
            'contact_matrix': list(list),
            'age_bin_edges': list,
            'age_distribution': list
        ...}

    """

    def __init__(self, fips, N_samples, t_list, I_initial=1, suppression_policy=None):

        # Caching globally to avoid relatively significant performance overhead
        # of loading for each county.
        super().__init__(fips, N_samples, t_list, I_initial, suppression_policy)
        global hosp_rate_data
        if hosp_rate_data is None:
            hosp_rate_data = load_data.load_cdc_hospitalization_data()
        self.contact_matrix_data = load_data.load_contact_matrix_data_by_fips(fips)
        self.population = np.array(self.contact_matrix_data[fips]["age_distribution"])

    def generate_age_specific_rates(self):
        """
        Generate age specific hospitalization_rate_general,
        hospitalization_rate_icu, and mortality_rate.

        Yields
        ------
          : np.array
            Following rates estimated by fitting age bin centers to the
            function interpolated using cdc hospitalization data:
            - hospitalization_rate_general
            - hospitalization_rate_icu
            - mortality_rate
        """
        age_bin_edges = self.contact_matrix_data[self.fips]["age_bin_edges"].copy()
        age_bin_edges.append(120)
        age_bin_centers = (
            np.array(age_bin_edges[1:]) + np.array(age_bin_edges[:-1])
        ) / 2

        for suffix in ["_hgen", "_icu", "_fatility"]:
            f = scipy.interpolate.interp1d(
                hosp_rate_data["lower_age"].tolist()
                + hosp_rate_data["mean_age"].tolist(),
                hosp_rate_data["lower%s" % suffix].tolist()
                + hosp_rate_data["mean%s" % suffix].tolist(),
            )
            yield f(age_bin_centers).clip(min=0)

    def generate_age_specific_initial_conditions(self):
        """
        Generate initial condition based on age distribution.

        Returns
        -------
        E_initial: np.array
            Array of zeros with shape (number of age bin edges, )
        A_initial: np.array
            Array of zeros with shape (number of age bin edges, )
        I_initial: np.array
            Array with shape (number of age bin edges, )
        HGen_initial: np.array
            Array of zeros with shape (number of age bin edges, )
        HICU_initial: np.array
            Array of zeros with shape (number of age bin edges, )
        HICUVent_initial: np.array
            Array of zeros with shape (number of age bin edges, )
        """
        age_dist = self.contact_matrix_data[self.fips]["age_distribution"]
        E_initial = np.zeros(len(age_dist))
        A_initial = np.zeros(len(age_dist))
        I_initial = self.I_initial * np.array(age_dist) / sum(age_dist)
        HGen_initial = np.zeros(len(age_dist))
        HICU_initial = np.zeros(len(age_dist))
        HICUVent_initial = np.zeros(len(age_dist))

        return (
            E_initial,
            A_initial,
            I_initial,
            HGen_initial,
            HICU_initial,
            HICUVent_initial,
        )

    def update_parameter_sets(self, parameter_sets):
        """
        Update sampled parameters to make them age-specific.

        Parameters
        ----------
        parameter_sets : list(dict)
             Parameters sampled for SEIR model without age structure.

        Returns
        -------
          :  list(dict)
             Parameter samples with age-specific parameters.
        """

        (
            E_initial,
            A_initial,
            I_initial,
            HGen_initial,
            HICU_initial,
            HICUVent_initial,
        ) = self.generate_age_specific_initial_conditions()

        (
            hospitalization_rate_general,
            hospitalization_rate_icu,
            mortality_rate,
        ) = self.generate_age_specific_rates()

        # rescale hospitalization rates to match the overall average
        hospitalization_rate_general = (
            hospitalization_rate_general * 0.04 / hospitalization_rate_general.mean()
        )
        hospitalization_rate_icu = (
            hospitalization_rate_icu * 0.04 * 0.3 / hospitalization_rate_icu.mean()
        )

        # shift to have mean 0.4
        mortality_rate_from_ICU = mortality_rate + 0.4 - mortality_rate.mean()

        contact_matrix = np.array(self.contact_matrix_data[self.fips]["contact_matrix"])
        age_bin_edges = np.array(self.contact_matrix_data[self.fips]["age_bin_edges"])

        for parameter_set in parameter_sets:
            # For now we have disabled this bucket and lowered rates of other
            # boxes accordingly. Since we were not modeling different contact
            # rates, this has the same result.

            # rescale to match overall average
            parameter_set.update(
                dict(
                    N=self.population,
                    A_initial=A_initial,
                    I_initial=I_initial,
                    E_initial=E_initial,
                    HGen_initial=HGen_initial,
                    HICU_initial=HICU_initial,
                    HICUVent_initial=HICUVent_initial,
                    age_bin_edges=age_bin_edges,
                    contact_matrix=np.random.normal(
                        loc=contact_matrix, scale=contact_matrix / 10
                    ).clip(min=0),
                    # These parameters produce an IFR ~0.0065 if we had infinite
                    # capacity, and about ~0.0125 with capacity constraints imposed
                    hospitalization_rate_general=np.random.normal(
                        loc=hospitalization_rate_general,
                        scale=hospitalization_rate_general / 10,
                    ).clip(min=0),
                    hospitalization_rate_icu=np.random.normal(
                        loc=hospitalization_rate_icu,
                        scale=hospitalization_rate_icu / 10,
                    ).clip(min=0),
                    # w/o ventilation, this would suggest a 20-42% mortality rate
                    # among general hospitalized patients w/o access to ventilators:
                    # “Among all patients, a range of 3% to 17% developed ARDS
                    # compared to a range of 20% to 42% for hospitalized patients
                    # and 67% to 85% for patients admitted to the ICU.1,4-6,8,11”
                    # 10% Of the population should die at saturation levels. CFR
                    # from Italy is 11.9% right now, Spain 8.9%.  System has to
                    # produce,
                    mortality_rate_from_ICU=np.random.normal(
                        loc=mortality_rate_from_ICU, scale=mortality_rate_from_ICU / 10
                    ).clip(min=0),
                )
            )

        return parameter_sets

    def sample_seir_parameters(self, override_params=None):
        """
        Generate N_samples of parameter values from the priors listed below.

        Parameters
        ----------
        override_params: dict()
            Individual parameters can be overridden here.

        Returns
        -------
        : list(dict)
            List of parameter sets to feed to the simulations.
        """
        override_params = override_params or dict()
        parameter_sets = super().sample_seir_parameters(override_params)
        parameter_sets = self.update_parameter_sets(parameter_sets)

        return parameter_sets

    def get_average_seir_parameters(self):
        """
        Sample from the ensemble to obtain the average parameter values.

        Returns
        -------
        average_parameters: dict
            Average of the parameter ensemble, determined by sampling.
        """
        df = pd.DataFrame(self.sample_seir_parameters()).drop(
            ["t_list", "suppression_policy"], axis=1
        )
        average_parameters = {}
        for col in df.columns:
            average_parameters[col] = df[col].mean()
        average_parameters["t_list"] = self.t_list
        average_parameters["suppression_policy"] = self.suppression_policy
        return average_parameters
