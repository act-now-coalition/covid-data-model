import unittest
import pandas as pd
import datetime
import time
import os.path
from libs.CovidTimeseriesModel import CovidTimeseriesModel
from libs.CovidDatasets import CDSDataset
from libs.CovidUtil import CovidUtil


class CovidTimeseriesModelTest(unittest.TestCase):
    r0 = 2.4
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

    def test_cds_dataset(self):
        self._test_model(CDSDataset(), self.interventions, 4, 'cds_data')

    def _test_model(self, dataset, interventions, model_interval, sub_dir):
        for state in dataset.get_all_states_by_country('USA'):
            print('Testing {}'.format(state))
            for i in range(0, len(interventions)):
                print(i)
                intervention = interventions[i]
                res = CovidUtil().model_us_state(state, dataset, model_interval, intervention).fillna('')
                o_fp = os.path.join('results', sub_dir, state + '.' + str(i) + '.csv')
                snapshot = pd.read_csv(o_fp, parse_dates=['Date']).fillna('')
                pd.testing.assert_frame_equal(res, snapshot, check_dtype=False)

    # TODO: Make it so that each state is a separate test, so if one fails we know which one.

if __name__ == '__main__':
    unittest.main()
