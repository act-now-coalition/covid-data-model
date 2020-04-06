#!/bin/bash

# Run State and County level models
./run_model.py state --deploy
./run_model.py county --deploy

# Generate the latest state case summary data.
./run_data.py latest --deploy
