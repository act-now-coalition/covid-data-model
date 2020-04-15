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
pyseir run-all --run-mode=can-before-hospitalization-new-params --output-dir="results/"

# Relocate output to the expected location.
cp results/web_ui/county/* ${PUBLIC_DATA_PATH}/county/
cp results/web_ui/state/* ${PUBLIC_DATA_PATH}/

# Previous method for invoking the original Python SEIR model follows.
#./run.py model state -o "${PUBLIC_DATA_PATH}"
#./run.py model county -o "${PUBLIC_DATA_PATH}/county"

# Generate demographic and case data summaries for counties.
./run.py model county-summary -o "${PUBLIC_DATA_PATH}/county_summaries"

# Generate the latest state case summary data.
./run.py data latest -o "${PUBLIC_DATA_PATH}/case_summary"
