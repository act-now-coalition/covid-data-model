import unittest
import pandas as pd
import datetime
import time
import os.path
from libs.CovidTimeseriesModel import CovidTimeseriesModel
from libs.CovidDatasets import CDSDataset, Dataset, JHUDataset
from libs.CovidUtil import CovidUtil

class CovidTimeseriesModelTest(unittest.TestCase):
    r0 = 2.4
    num_snapshot_tests = 50
    interventions = [
        None,
        {
            datetime.date(2020, 3, 23): 1.3,
            datetime.date(2020, 4, 20): 1.1,
            datetime.date(2020, 5, 22): 0.8,
            datetime.date(2020, 6, 23): r0
        },
        {
            datetime.date(2020, 3, 23): 1.7,
            datetime.date(2020, 6, 23): r0
        },
        {
            datetime.date(2020, 3, 23): 1.3,
            datetime.date(2020, 3, 31): 0.3,
            datetime.date(2020, 4, 28): 0.2,
            datetime.date(2020, 5, 6): 0.1,
            datetime.date(2020, 5, 10): 0.35,
            datetime.date(2020, 5, 18): r0
        }
    ]

    def test_calculate_r(self):
        r0 = 0
        current_cycle = {'confirmed': 5}
        previous_cycle = {'confirmed': 0}
        self.assertAlmostEqual(
            CovidTimeseriesModel().calculate_r(current_cycle, previous_cycle, r0),
            r0
        )

        current_cycle = {'confirmed': 15}
        previous_cycle = {'confirmed': 10}
        self.assertAlmostEqual(
            CovidTimeseriesModel().calculate_r(current_cycle, previous_cycle, 0),
            15/10
        )

    def test_model(self):
        """Compare the output of the model in it's current state to the outputs it's had in the past"""
        self._compare_snapshot_tests(
            self._build_test_parameter_sets(),
            self._get_snapshots()
        )


    def _build_test_parameter_sets(self):
        dataset = CDSDataset()
        states = dataset.get_all_states_by_country('USA')
        MODEL_INTERVAL = 4
        sets = []
        for state in states[:2]:
            for intervention in self.interventions:
                sets.append(
                    CovidUtil().initialize_model_parameters({
                        # Pack the changeable model parameters
                        'timeseries': dataset.get_timeseries_by_country_state('USA', state, MODEL_INTERVAL),
                        'beds': dataset.get_beds_by_country_state('USA', state),
                        'population': dataset.get_population_by_country_state('USA', state),
                        'projection_iterations': 25,  # Number of iterations into the future to project
                        'r0': 2.4,
                        'interventions': intervention,
                        'hospitalization_rate': .0727,
                        'initial_hospitalization_rate': .05,
                        'case_fatality_rate': .0109341104294479,
                        'hospitalized_cases_requiring_icu_care': .1397,
                        # Assumes that anyone who needs ICU care and doesn't get it dies
                        'hospital_capacity_change_daily_rate': 1.05,
                        'max_hospital_capacity_factor': 2.07,
                        'initial_hospital_bed_utilization': .6,
                        'model_interval': MODEL_INTERVAL,  # In days
                        'total_infected_period': 12,  # In days
                    })
                )
        return sets

    def _compare_snapshot_tests(self, model_params, snapshots):
        """Compare the output of the model in its current state to the outputs it had at a previous vetted state"""
        for i in range(0, len(model_params)):
            with self.subTest(state=model_params[i]['state'], i=i):
                # SubTest will make it easier to tell which snapshot failed, if any
                pd.testing.assert_frame_equal(
                    CovidTimeseriesModel().forecast(model_params[i]),
                    snapshots[i],
                    check_dtype=False
                )

    def _update_snapshots(self):
        """Update the snapshots"""
        with open(r'test/snapshots/snapshots.json', 'w') as out:
            for snap in [
                df.to_json() for df in [
                    CovidTimeseriesModel().forecast(p)
                    for p in self._build_test_parameter_sets()
                ]
            ]:
                out.write(snap)
                out.write('\n')

    def _get_snapshots(self):
        """Read the snapsnot file into memory and construct the DataFrames"""
        snaps = []
        with open(r'snapshots/snapshots.json') as snaps_file:
            for s in snaps_file:
                # We don't want
                snap = pd.read_json(s, convert_dates=False)
                # read_json doesn't let us specify the columns to parse as dates, so we have to do it ourselves
                snap['Date'] = pd.to_datetime(snap['Date'], unit='ms')
                snaps.append(snap)
        return snaps


if __name__ == '__main__':
    unittest.main()
