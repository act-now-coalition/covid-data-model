#!/bin/bash

PUBLIC_DATA_PATH=../covid-projections/public/data


# Run State and County level models
./run_model.py state -o ${PUBLIC_DATA_PATH}
./run_model.py county -o ${PUBLIC_DATA_PATH}/county
./run_model.py county-summary -o ${PUBLIC_DATA_PATH}/county_summaries
# Generate the latest state case summary data.
./run_data.py latest -o ${PUBLIC_DATA_PATH}/case_summary
