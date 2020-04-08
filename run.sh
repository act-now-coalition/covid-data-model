#!/bin/bash
# run.sh - Runs everything necessary to generate our API artifacts (for
# website, external consumers, etc.) based on our inputs (from
# covid-data-public).

set -o nounset
set -o errexit

# Checks command-line arguments, sets variables, etc.
prepare () {
  # Go to repo root (where run.sh lives).
  cd "$(dirname "$0")"

  # Parse args if specified.
  if [ $# -ne 2 ]; then
    echo "Usage: $0 [covid-data-public directory] [output-directory]"
    echo
    echo "Example: $0 ../covid-data-public/ ./api_results/"
    exit 1
  else
    DATA_SOURCES_DIR="$(absPath $1)"
    API_OUTPUT_DIR="$(absPath $2)"
  fi

  if [ ! -d "${DATA_SOURCES_DIR}" ] ; then
    echo "Directory ${DATA_SOURCES_DIR} does not exist."
    exit 1
  fi

  if [ ! -d "${API_OUTPUT_DIR}" ] ; then
    echo "Directory ${API_OUTPUT_DIR} does not exist. Creating."
    mkdir -p "${API_OUTPUT_DIR}"
  fi

  # run_model.py uses the COVID_DATA_PUBLIC environment variable to find inputs.
  export COVID_DATA_PUBLIC="${DATA_SOURCES_DIR}"

  # These directiories essentially define the structure of our API endpoints.
  # TODO: These should perhaps live in python, near the schemas (defined in api/)?
  STATES_DIR="${API_OUTPUT_DIR}/";
  # TODO: I think deploy_dod_dataset.py may currently have an implicit
  # requirement that the county model JSON is in a /county subdirectory of the
  # states?
  COUNTIES_DIR="${API_OUTPUT_DIR}/county";
  COUNTY_SUMMARIES_DIR="${API_OUTPUT_DIR}/county_summaries";
  CASE_SUMMARIES_DIR="${API_OUTPUT_DIR}/case_summary"

  # TODO: Move DoD onto a more general-purpose schema rather than treat them custom.
  DOD_DIR="${API_OUTPUT_DIR}/custom1"
}

execute() {
  echo ">>> Generating state models to ${STATES_DIR}"
  # TODO(#148): We need to clean up the output of these scripts!
  ./run_model.py state -o "${API_OUTPUT_DIR}" > /dev/null

  echo ">>> Generating county models to ${COUNTIES_DIR}"
  # TODO(#148): We need to clean up the output of these scripts!
  ./run_model.py county -o "${COUNTIES_DIR}" > /dev/null

  echo ">>> Generating county summaries to ${COUNTY_SUMMARIES_DIR}"
  # TODO(#148): We need to clean up the output of these scripts!
  ./run_model.py county-summary -o "${COUNTY_SUMMARIES_DIR}" > /dev/null

  echo ">>> Generating case summaries to ${CASE_SUMMARIES_DIR}"
  # TODO(#148): We need to clean up the output of these scripts!
  ./run_data.py latest -o "${CASE_SUMMARIES_DIR}" > /dev/null

  echo ">>> Generating DoD artifacts to ${DOD_DIR}"
  mkdir -p "${DOD_DIR}"
  ./deploy_dod_dataset.py -i "${STATES_DIR}" -o "${DOD_DIR}"

  echo ">>> All API Artifacts written to ${API_OUTPUT_DIR}"
}

# Helper for getting absolute paths.
function absPath() {
  (
  cd "$(dirname $1)"
  echo "$PWD/$(basename $1)"
  )
}

prepare "$@"
execute
