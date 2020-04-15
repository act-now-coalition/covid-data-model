#!/bin/bash
# label-api.sh - Assigns a label (e.g. 'v0') to a published API snapshot (e.g '15')

set -o nounset
set -o errexit

CMD=$0

# Checks command-line arguments, sets variables, etc.
prepare () {
  # Parse args if specified.
  if [ $# -ne 2 ]; then
    exit_with_usage
  else
    LABEL=$1
    SNAPSHOT_ID=$2
  fi

  if ! [[ $SNAPSHOT_ID =~ ^[0-9]+$ ]] ; then
    echo "Error: Specified Snapshot ID ($SNAPSHOT_ID) should be a plain number."
    echo
    exit_with_usage
  fi

  if [[ $LABEL =~ ^/ ]] ; then
    echo "Error: Specified Label ($LABEL) should not start with a '/'."
    echo
    exit_with_usage
  fi

  if [[ -z ${GITHUB_TOKEN:-} ]]; then
    echo "Error: GITHUB_TOKEN must be set to a personal access token. See:"
    echo "https://help.github.com/en/github/authenticating-to-github/creating-a-personal-access-token-for-the-command-line"
    exit 1
  fi
}

exit_with_usage () {
  echo "Usage: $CMD <label> <snapshot-id>"
  echo
  echo "Examples:"
  echo "$CMD v0 15                # Points /v0 at /snapshot/15"
  echo "$CMD snapshot/latest 15   # Points /snapshot/latest at /snapshot/15"
  exit 1
}

execute () {
  curl -H "Authorization: token $GITHUB_TOKEN" \
      --request POST \
      --data "{\"event_type\": \"label-api-snapshot\", \"client_payload\": { \"label\": \"${LABEL}\", \"snapshot_id\": \"${SNAPSHOT_ID}\" } }" \
      https://api.github.com/repos/covid-projections/covid-data-model/dispatches

  echo "Label requested. Go to https://github.com/covid-projections/covid-data-model/actions to monitor progress."
}

prepare "$@"
execute
