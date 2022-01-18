
#!/bin/bash
# run.sh - Runs everything necessary to generate our API artifacts (for
# website, external consumers, etc.) based on our inputs (from
# covid-data-public).

set -o nounset
set -o errexit

# Checks command-line arguments, sets variables, etc.
prepare () {
  # Parse args if specified.
  if [ $# -lt 2 ] || [ $# -gt 3 ]; then
    echo "Usage: $0 [covid-data-public directory] [output-directory] (optional - specific function)"
    echo
    echo "Example: $0 ../covid-data-public/ ./api-results/"
    echo "Example: $0 ../covid-data-public/ ./api-results/ execute_model"
    echo "Example: $0 ../covid-data-public/ ./api-results/ execute_api"
    exit 1
  else
    DATA_SOURCES_DIR="$(abs_path $1)"
    API_OUTPUT_DIR="$(abs_path $2)"
    API_OUTPUT_V2="${API_OUTPUT_DIR}/v2"

    echo $DATA_SOURCES_DIR
    echo $API_OUTPUT_DIR
  fi

  if [ $# -eq 2 ]; then
    EXECUTE_FUNC="execute"
  else
    EXECUTE_FUNC="${3}"
  fi

  if [ ! -d "${DATA_SOURCES_DIR}" ] ; then
    echo "Directory ${DATA_SOURCES_DIR} does not exist."
    exit 1
  fi

  if [ ! -d "${API_OUTPUT_DIR}" ] ; then
    echo "Directory ${API_OUTPUT_DIR} does not exist. Creating."
    mkdir -p "${API_OUTPUT_DIR}"
    echo "made dir"
  fi

  if [ ! -d "${API_OUTPUT_V2}" ] ; then
    echo "Directory ${API_OUTPUT_V2} does not exist. Creating."
    mkdir -p "${API_OUTPUT_V2}"
    echo "made dir"
  fi


  # run_model.py uses the COVID_DATA_PUBLIC environment variable to find inputs.
  export COVID_DATA_PUBLIC="${DATA_SOURCES_DIR}"

  # These directiories essentially define the structure of our API endpoints.
  # TODO: These should perhaps live in python, near the schemas (defined in api/)?
  API_OUTPUT_COUNTIES="${API_OUTPUT_DIR}/us/counties"
  API_OUTPUT_STATES="${API_OUTPUT_DIR}/us/states"
  API_OUTPUT_US="${API_OUTPUT_DIR}/us"
  API_OUTPUT_QA="${API_OUTPUT_DIR}/qa"
  # Create QA dir
  if [ ! -d "${API_OUTPUT_QA}" ] ; then
    echo "Directory ${API_OUTPUT_QA} does not exist. Creating."
    mkdir -p "${API_OUTPUT_QA}"
    echo "made dir"
  fi

  SOURCE_DATA_DIR="./data"
}

execute_raw_data_qa() {
  # Go to repo root (where run.sh lives).
  RAW_DATA_OUTPUT_STREAM="/RAW_QA"
  cd "$(dirname "$0")"
  rm -rf "${API_OUTPUT_DIR}${RAW_DATA_OUTPUT_STREAM}"
  python ./raw_data_QA/check_raw_case_death_data.py --output_dir="${API_OUTPUT_DIR}${RAW_DATA_OUTPUT_STREAM}" --covid_data_public_dir="${DATA_SOURCES_DIR}"
}

execute_model() {
  # Go to repo root (where run.sh lives).
  cd "$(dirname "$0")"

  echo ">>> Generating state and county models to ${API_OUTPUT_DIR}"
  # TODO(#148): We need to clean up the output of these scripts!
  pyseir build-all --output-dir="${API_OUTPUT_DIR}" | tee "${API_OUTPUT_DIR}/stdout.log"

  # Move state output to the expected location.
  mkdir -p ${API_OUTPUT_DIR}/

  # Capture all the PDFs pyseir creates in output/pyseir since they are
  # extremely helpful for debugging / QA'ing the model results.
  echo ">>> Generating pyseir.zip from PDFs in output/pyseir."
  pushd output
  zip -r "${API_OUTPUT_DIR}/pyseir.zip" pyseir/* -i '*.pdf'
  popd
}

execute_api() {
  # Go to repo root (where run.sh lives).
  cd "$(dirname "$0")"

  echo ">>> Generating ${API_OUTPUT_DIR}/version.json"
  generate_version_json "${API_OUTPUT_DIR}"

  echo ">>> Generating API for states to ${API_OUTPUT_STATES}/{STATE_ABBREV}.{INTERVENTION}.json"
  mkdir -p "${API_OUTPUT_STATES}"
  ./run.py api generate-api  -i "${API_OUTPUT_DIR}" -o "${API_OUTPUT_STATES}" --summary-output "${API_OUTPUT_US}" -l state

  echo ">>> Generating API for counties to ${API_OUTPUT_COUNTIES}/{FIPS}.{INTERVENTION}.json"
  ./run.py api generate-api  -i "${API_OUTPUT_DIR}" -o "${API_OUTPUT_COUNTIES}" --summary-output "${API_OUTPUT_US}" -l county

  # echo ">>> Generate an QA doc for states to ${API_OUTPUT_DIR}/qa"
  # ./run.py compare-snapshots -i "${API_OUTPUT_STATES}" -o "${API_OUTPUT_DIR}/qa"

  echo ">>> Copying source data (and summary, provenance, etc. reports) to ${API_OUTPUT_QA}"
  cp -r "${SOURCE_DATA_DIR}"/* "${API_OUTPUT_QA}"

  echo ">>> All API Artifacts written to ${API_OUTPUT_DIR}"
}


execute_api_v2() {
  # Go to repo root (where run.sh lives).
  cd "$(dirname "$0")"

  echo ">>> Generating ${API_OUTPUT_V2}/version.json"
  generate_version_json "${API_OUTPUT_V2}"

  echo ">>> Generating API Output"
  ./run.py api generate-api-v2 "${API_OUTPUT_DIR}" -o "${API_OUTPUT_V2}"

  echo ">>> All API Artifacts written to ${API_OUTPUT_V2}"
}


execute_zip_folder() {
  # Go to repo root (where run.sh lives).
  cd "$(dirname "$0")"

  echo ">>> Generating all.zip with all API artifacts."
  #pushd "${API_OUTPUT_DIR}/.."
  API_RESULTS_ZIP="${API_OUTPUT_DIR}/api-results.zip"
  zip -r ${API_RESULTS_ZIP} "${API_OUTPUT_DIR}/"
  #popd
}


execute() {
  execute_model
  execute_api_v2
  execute_api
  execute_zip_folder
}

### Utilities for scripting

# Generates a version.json file in the API_OUTPUT_DIR capturing the time
# and state of all repos.
function generate_version_json() {
  local api_output_dir="$1"
  local model_repo_json=$(get_repo_status_json)

  local timestamp=$(iso_timestamp)

  cat > "${api_output_dir}/version.json" << END
{
  "timestamp": "${timestamp}",
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

case $EXECUTE_FUNC in
  execute_model)
    echo "Executing Model Results"
    execute_model
    ;;
  execute_api)
    echo "Executing Api"
    execute_api
    ;;
  execute_api_v2)
    echo "Executing Api V2"
    execute_api_v2
    ;;
  execute_raw_data_qa)
    echo "Executing Raw Data QA"
    execute_raw_data_qa
    ;;
  execute_zip_folder)
    echo "Executing Api"
    execute_zip_folder
    ;;
  execute)
    echo "Executing Entire Pipeline"
    execute
    ;;
  *)
    echo "Invalid Function. Exiting"
    ;;
esac
