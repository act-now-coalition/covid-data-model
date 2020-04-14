#!/bin/bash
# run.sh - Runs everything necessary to generate our API artifacts (for
# website, external consumers, etc.) based on our inputs (from
# covid-data-public).

set -o nounset
set -o errexit

# Checks command-line arguments, sets variables, etc.
prepare () {
  # Parse args if specified.
  if [ $# -ne 2 ]; then
    echo "Usage: $0 [covid-data-public directory] [output-directory]"
    echo
    echo "Example: $0 ../covid-data-public/ ./api_results/"
    exit 1
  else
    DATA_SOURCES_DIR="$(abs_path $1)"
    API_OUTPUT_DIR="$(abs_path $2)"
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

  INPUT_BASE_DIR="${API_OUTPUT_DIR}/";

  API_OUTPUT_COUNTIES = "${API_OUTPUT_DIR}/us/counties"
  API_OUTPUT_STATES = "${API_OUTPUT_DIR}/us/states"

  # TODO: I think deploy_dod_dataset.py may currently have an implicit
  # requirement that the county model JSON is in a /county subdirectory of the
  # states?
  COUNTY_SUMMARIES_DIR="${API_OUTPUT_DIR}/county_summaries";
  CASE_SUMMARIES_DIR="${API_OUTPUT_DIR}/case_summary"

  # TODO: Move DoD onto a more general-purpose schema rather than treat them custom.
  DOD_DIR="${API_OUTPUT_DIR}/custom1"
}

execute() {
  # Go to repo root (where run.sh lives).
  cd "$(dirname "$0")"

  echo ">>> Generating state models to ${API_OUTPUT_DIR}"
  # TODO(#148): We need to clean up the output of these scripts!
  ./run.py model state -o "${API_OUTPUT_DIR}" > /dev/null

  echo ">>> Generating county models to ${API_OUTPUT_DIR}/county"
  # TODO(#148): We need to clean up the output of these scripts!
  ./run.py model county -o "${API_OUTPUT_DIR}/county" > /dev/null

  echo ">>> Generating county summaries to ${COUNTY_SUMMARIES_DIR}"
  # TODO(#148): We need to clean up the output of these scripts!
  ./run.py model county-summary -o "${COUNTY_SUMMARIES_DIR}" > /dev/null

  echo ">>> Generating case summaries to ${CASE_SUMMARIES_DIR}"
  # TODO(#148): We need to clean up the output of these scripts!
  ./run.py data latest -o "${CASE_SUMMARIES_DIR}" > /dev/null

  echo ">>> Generating DoD artifacts to ${DOD_DIR}"
  mkdir -p "${DOD_DIR}"
  ./run.py deploy-dod -i "${INPUT_BASE_DIR}" -o "${DOD_DIR}"

  echo ">>> Generating ${API_OUTPUT_DIR}/version.json"
  generate_version_json

  echo ">>> Generating Top 100 Counties json to ${API_OUTPUT_COUNTIES}/counties_top_100.json"
  mkdir -p "${API_OUTPUT_COUNTIES}"
  ./run.py deploy-top-counties -i "${INPUT_BASE_DIR}" -o "${API_OUTPUT_COUNTIES}"

  echo ">>> Generating API for states to ${API_OUTPUT_STATES}/{STATE_ABBREV}.{INTERVENTION}.json"
  mkdir -p "${API_OUTPUT_STATES}"
  ./run.py deploy-states-api -i "${INPUT_BASE_DIR}" -o "${API_OUTPUT_STATES}"

  echo ">>> Generating API for states to ${API_OUTPUT_COUNTIES}/{FIPS}.{INTERVENTION}.json"
  ./run.py deploy-counties-api -i "${INPUT_BASE_DIR}" -o "${API_OUTPUT_COUNTIES}"

  echo ">>> All API Artifacts written to ${API_OUTPUT_DIR}"
}

# Generates a version.json file in the API_OUTPUT_DIR capturing the time
# and state of all repos.
function generate_version_json() {
  local model_repo_json=$(get_repo_status_json)

  pushd "${DATA_SOURCES_DIR}" > /dev/null
  local data_repo_json=$(get_repo_status_json)
  popd > /dev/null

  local timestamp=$(iso_timestamp)

  cat > "${API_OUTPUT_DIR}/version.json" << END
{
  "timestamp": "${timestamp}",
  "covid-data-public": ${data_repo_json},
  "covid-data-model": ${model_repo_json}
}
END
}

# Returns a { branch: ..., hash: ..., dirty: ... } JSON blob describing the
# status of the repo we are currently in.
function get_repo_status_json() {
  local branch=$(get_repo_branch)
  local hash=$(get_repo_commit_hash)
  local dirty=$(get_repo_dirty)

  echo "{ \"branch\": \"${branch}\", \"hash\": \"${hash}\", \"dirty\": ${dirty} }"
}

# Returns the name of the active branch or just "HEAD" if it's detached.
function get_repo_branch() {
  local branch_name=$(git symbolic-ref -q HEAD)
  branch_name=${branch_name##refs/heads/}
  branch_name=${branch_name:-HEAD}
  echo "${branch_name}"
}

# Returns the SHA hash of the latest commit.
function get_repo_commit_hash() {
  git rev-parse --verify HEAD
}

# Returns true if there are any modified / untracked files in the repo.
function get_repo_dirty() {
  if [ -z "$(git status --porcelain)" ]; then
    echo "false"
  else
    echo "true"
  fi
}

# Returns a UTC timestamp in ISO 8601 format.
function iso_timestamp() {
  date -u +%Y-%m-%dT%H:%M:%S%z
}

# Helper for getting absolute paths.
function abs_path() {
  (
  cd "$(dirname $1)"
  echo "$PWD/$(basename $1)"
  )
}

prepare "$@"
execute
