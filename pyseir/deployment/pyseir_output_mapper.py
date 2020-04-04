import os
from pyseir import OUTPUT_DIR
from pyseir import load_data


class PyseirOutputMapper:

    def __init__(self, state, output_interval_days=4):
        """

        Parameters
        ----------
        state: str
            State to map outputs for.
        output_interval_days: int
            Outputs data interval in days. Pyseir outputs are downsampled via
            linear interpolation to this level.
        """

        self.output_cols = [
            "date",
            "a",  # Dummy
            "b",  # Dummy
            "c",  # Dummy
            "d",  # Dummy
            "e",  # Dummy
            "f",  # Dummy
            "g",  # Dummy
            "all_hospitalized",
            "all_infected",
            "dead",
            "beds",
            "i",  # Dummy
            "j",  # Dummy
            "k",  # Dummy
            "l",  # Dummy
            "population",
            "m",  # Dummy
            "n",  # Dummy
        ]

    def map_fips(self, fips):
        """
        For a given county fips code, generate the CAN UI output format.

        Parameters
        ----------
        fips

        Returns
        -------

        """
        county_metadata = load_data.load_county_metadata_by_fips(fips)
        pyseir_outputs = load_data.load_ensemble_results(fips)

        for suppression_policy, output in pyseir_outputs.items():

