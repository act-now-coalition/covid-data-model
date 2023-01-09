#!/bin/bash
# label-api.sh - Assigns the label 'latest' to a published API snapshot (e.g '15')

set -o nounset
set -o errexit

CMD=$0

# Checks command-line arguments, sets variables, etc.
prepare () {
  # Parse args if specified.
  if [ $# -ne 1 ]; then
    exit_with_usage
  else
    SNAPSHOT_ID=$1
  fi

  if ! [[ $SNAPSHOT_ID =~ ^[0-9]+$ ]] ; then
    echo "Error: Specified Snapshot ID ($SNAPSHOT_ID) should be a plain number."
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
  echo "Usage: $CMD <snapshot-id>"
  echo
  echo "Examples:"
  echo "$CMD 370   # Points /latest at /snapshot/370"
  exit 1
}

execute () {
  curl -H "Authorization: token $GITHUB_TOKEN" \
      --request POST \
      --data "{ \"ref\": \"main\", \"inputs\": { \"snapshot_id\": \"${SNAPSHOT_ID}\" } }" \
      https://api.github.com/repos/act-now-coalition/covid-data-model/actions/workflows/label_api_snapshot.yml/dispatches

  echo "Label requested. Go to https://github.com/act-now-coalition/covid-data-model/actions to monitor progress."
}

prepare "$@"
execute
