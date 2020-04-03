import logging
import argparse

from libs.build_params import OUTPUT_DIR, get_interventions
from datetime import datetime, timedelta
from libs.datasets import JHUDataset
from libs.datasets.dataset_utils import AggregationLevel
import os.path
import simplejson
from libs.us_state_abbrev import us_state_abbrev, us_fips
from libs.build_dod_dataset import get_usa_by_county_df, get_usa_by_states_df, get_usa_county_shapefile, get_usa_state_shapefile
from libs.build_dod_dataset import get_projections_df
"""
1) 15 day hosp, 30 day hosp, 15 day bed gap, 30 day bed gap 
2)  # Indexes used by website JSON:
    # date: 0,
    # hospitalizations: 8,
    # cumulativeInfected: 9,
    # cumulativeDeaths: 10,
    # beds: 11,
    # totalPopulation: 16,

3) Get the 15 days, 30 date
4) for each state look for that date
5) save that date and print it out in some format
"""

# def get_hospitals_and_shortfalls(projection, days_out): 
#     for row in projection:
#         row_time = datetime.strptime(row[1], '%m/%d/%y')

#         if row_time >= days_out:
#             hospitilzations = int(row[9])
#             beds = int(row[12])
#             short_fall = abs(hospitilzations - beds) if hospitilzations > beds else 0
#             return hospitilzations, short_fall
#     return 0, 0

# def get_projections_df():
#     # for each state in our data look at the results we generated via run.py 
#     # to create the projections
#     intervention_type = 0 # None, as requested

#     # get 16 and 32 days out from now
#     dt = datetime.now()
#     sixteen_days = dt + timedelta(days=16)
#     thirty_two_days = dt + timedelta(days=32)

#     #save results in a list of lists, converted to df later
#     results = []

#     for state in list(us_state_abbrev.values()):
#         file_name = f"{state}.{intervention_type}.json"
#         path = os.path.join(OUTPUT_DIR, file_name)

#         # if the file exists in that directory then process
#         if os.path.exists(path):
#             with open(path, "r") as projections:
#                 projection =  simplejson.load(projections)

#                 hosp_16_days, short_fall_16_days = get_hospitals_and_shortfalls(projection, sixteen_days)
#                 hosp_32_days, short_fall_32_days = get_hospitals_and_shortfalls(projection, thirty_two_days)

#                 results.append([state, hosp_16_days, hosp_32_days, short_fall_16_days, short_fall_32_days])
#     print(results)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(get_projections_df())