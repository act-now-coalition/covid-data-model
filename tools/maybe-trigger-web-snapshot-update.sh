#!/bin/bash
# maybe-trigger-web-snapshot-update.sh
#
# Helper script called at the end of the deploy_api.yml workflow to potentially
# trigger the update-snapshot.yml workflow in the covid-projections repo to
# update to the newly generated snapshot.

set -o nounset
set -o errexit

CMD=$0

# The covid-projections branch to open a PR against.
BRANCH="develop"

# Checks command-line arguments, sets variables, etc.
prepare () {
  # Parse args if specified.
  if [ $# -ne 3 ]; then
    exit_with_usage
  else
    SNAPSHOT_ID=$1
    COVID_DATA_MODEL_REF=$2
    COVID_DATA_PUBLIC_REF=$3
  fi

  if ! [[ $SNAPSHOT_ID =~ ^[0-9]+$ ]] ; then
    echo "Error: Specified Snapshot ID ($SNAPSHOT_ID) should be a plain number."
    echo
    exit_with_usage
  fi

  if [[ $COVID_DATA_MODEL_REF != "master" ]] || [[ $COVID_DATA_PUBLIC_REF != "master" ]]; then
    echo "Not triggering covid-projections update-snapshot since this isn't a 'master' branch run."
    exit 0
  fi

  # We have daily jobs scheduled at 00:30 and 12:30 GMT. We want to run after the 12:30 one.
  currenttime=$(date -u +%H:%M)
  if [[ "$currenttime" < "12:00" ]]; then
    echo "Not triggering covid-projections update-snapshot since time (${currenttime}) is before 12:00 GMT."
    exit 0
  fi

  if [[ -z ${GITHUB_TOKEN:-} ]]; then
    echo "Error: GITHUB_TOKEN must be set to a personal access token. See:"
    echo "https://help.github.com/en/github/authenticating-to-github/creating-a-personal-access-token-for-the-command-line"
    exit 1
  fi
}

exit_with_usage () {
  echo "Usage: $CMD <snapshot-id> <covid-data-model-ref> <covid-data-public-ref>"
  echo
  echo "Examples:"
  echo "$CMD 123 master master"
  exit 1
}

execute () {
  curl -H "Authorization: token $GITHUB_TOKEN" \
      --request POST \
      --data "{\"event_type\": \"update-data-snapshot\", \"client_payload\": { \"branch\": \"${BRANCH}\", \"snapshot_id\": \"${SNAPSHOT_ID}\" } }" \
      https://api.github.com/repos/covid-projections/covid-projections/dispatches

  echo "Snapshot update requested. Go to https://github.com/covid-projections/covid-projections/actions to monitor progress."
}

prepare "$@"
execute
