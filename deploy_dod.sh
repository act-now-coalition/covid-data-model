#!/bin/bash
# deploy_dod.sh - Deploy DoD assets.
#
# By default writes to dod_results/ directory.
# Set BUCKET_NAME env var to deploy to S3 instead.

set -o nounset
set -o errexit

# Go to repo root.
cd "$(dirname "$0")"

# Directory for intermediate model files used by deploy_dod_dataset.py.
MODELS_DIR="results"
RESULTS_DIR="results/dod_results"
mkdir -p "${MODELS_DIR}"

# Run State and County level models
pyseir run-all --run-mode=can-before-hospitalization --output-dir="${MODELS_DIR}"
#./run.py model state -o "${MODELS_DIR}/state"
#./run.py model county -o "${MODELS_DIR}/county"

mkdir -p dod_results
./run.py deploy-dod -i "${MODELS_DIR}/web_ui" -o "${RESULTS_DIR}"
