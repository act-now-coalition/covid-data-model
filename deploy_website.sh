#!/bin/bash
# deploy_website.sh - Deploy model assets to website (../covid-projections)

set -o nounset
set -o errexit

# Go to repo root.
cd "$(dirname "$0")"

PUBLIC_DATA_PATH="../covid-projections/public/data"
if [ ! -d "${PUBLIC_DATA_PATH}" ] ; then
  echo "Directory ${PUBLIC_DATA_PATH} does not exist. Make sure you've cloned covid-projections next to covid-data-model."
  exit 1
fi

# Run State and County level models
./run_model.py state -o "${PUBLIC_DATA_PATH}"
./run_model.py county -o "${PUBLIC_DATA_PATH}/county"
./run_model.py county-summary -o "${PUBLIC_DATA_PATH}/county_summaries"
# Generate the latest state case summary data.
./run_data.py latest -o "${PUBLIC_DATA_PATH}/case_summary"
