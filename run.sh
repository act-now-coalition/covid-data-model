#!/bin/bash
# run.sh - Runs everything necessary to generate our API artifacts (for
# website, external consumers, etc.)

set -o nounset
set -o errexit

# Checks command-line arguments, sets variables, etc.
prepare () {
  # Parse args if specified.
  if [ $# -lt 1 ] || [ $# -gt 3 ]; then
    echo "Usage: $0 [output-directory] (optional - specific function) (optional - pyseir model snapshot number)"
    echo
    echo "Example: $0 ./api-results/"
    echo "Example: $0 ./api-results/ execute_model"
    echo "Example: $0 ./api-results/ execute_api_v2"
    echo "Example: $0 ./api-results/ execute_model 2920"
    exit 1
  else
    API_OUTPUT_DIR="$(abs_path $1)"
    API_OUTPUT_V2="${API_OUTPUT_DIR}/v2"

    echo $API_OUTPUT_DIR
  fi

  if [ $# -eq 1 ]; then
    EXECUTE_FUNC="execute"
  else
    EXECUTE_FUNC="${2}"
  fi

  if [ $# -ge 3 ]; then
    PYSEIR_ARTIFACT_SNAPSHOT="$3"
  else
    PYSEIR_ARTIFACT_SNAPSHOT=""
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

  # These directiories essentially define the structure of our API endpoints.
  # TODO: These should perhaps live in python, near the schemas (defined in api/)?
  API_OUTPUT_COUNTIES="${API_OUTPUT_DIR}/us/counties"
  API_OUTPUT_STATES="${API_OUTPUT_DIR}/us/states"
  API_OUTPUT_US="${API_OUTPUT_DIR}/us"

  SOURCE_DATA_DIR="./data"
}

execute_model() {
  # Go to repo root (where run.sh lives).
  cd "$(dirname "$0")"

  if [ ! -z "$PYSEIR_ARTIFACT_SNAPSHOT" ]; then
    echo ">>> Downloading state and county models from existing snapshot ${PYSEIR_ARTIFACT_SNAPSHOT}."
    # TODO(sean): Might be simpler to just download the model results to /tmp/
    API_OUTPUT_PARENT="${API_OUTPUT_DIR%/*}"
    ./run.py utils download-model-artifact "${PYSEIR_ARTIFACT_SNAPSHOT}" --output-dir=${API_OUTPUT_PARENT}

    echo ">>> Moving downloaded models to the expected locations."
    rm -r "${API_OUTPUT_DIR}"  # remove the empty directories created above
    mv "${API_OUTPUT_PARENT}"/api-results-${PYSEIR_ARTIFACT_SNAPSHOT} "${API_OUTPUT_DIR}"
  else
    echo ">>> Generating state and county models to ${API_OUTPUT_DIR}"
    # TODO(#148): We need to clean up the output of these scripts!
    python pyseir/cli.py build-all --output-dir="${API_OUTPUT_DIR}" | tee "${API_OUTPUT_DIR}/stdout.log"


    # Move state output to the expected location.
    mkdir -p ${API_OUTPUT_DIR}/

    # Capture all the PDFs pyseir creates in output/pyseir since they are
    # extremely helpful for debugging / QA'ing the model results.
    echo ">>> Generating pyseir.zip from PDFs in output/pyseir."
    pushd output
    zip -r "${API_OUTPUT_DIR}/pyseir.zip" pyseir/* -i '*.pdf'
    popd
  fi
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
  API_RESULTS_ZIP="${API_OUTPUT_DIR}/api-results.zip"
  zip -r ${API_RESULTS_ZIP} "${API_OUTPUT_DIR}/"
}


execute() {
  execute_model
  execute_api_v2
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
  execute_api_v2)
    echo "Executing Api V2"
    execute_api_v2
    ;;
  execute_zip_folder)
    echo "Executing zip folder"
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
