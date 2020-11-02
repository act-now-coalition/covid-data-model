#!/bin/bash
# maybe-trigger-label-api.sh
#
# Helper script called at the end of the deploy_api.yml workflow to potentially
# trigger the label-api workflow.

set -o nounset
set -o errexit

CMD=$0


# Checks command-line arguments, sets variables, etc.
prepare () {
  # Parse args if specified.
  if [ $# -ne 2 ]; then
    exit_with_usage
  else
    SNAPSHOT_ID=$1
    COVID_DATA_MODEL_REF=$2
    DIR=$(dirname "$0")
  fi

  if ! [[ $SNAPSHOT_ID =~ ^[0-9]+$ ]] ; then
    echo "Error: Specified Snapshot ID ($SNAPSHOT_ID) should be a plain number."
    echo
    exit_with_usage
  fi

  if [[ $COVID_DATA_MODEL_REF != "master" ]]; then
    echo "Not triggering label-api since this isn't a 'master' branch run."
    exit 0
  fi

  # TODO(chris) Comment: Should we trigger update regardless?
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
  echo "Usage: $CMD <snapshot-id> <covid-data-model-ref>"
  echo
  echo "Examples:"
  echo "$CMD 123 master"
  exit 1
}

execute () {
  $DIR/label-api.sh $SNAPSHOT_ID
}

prepare "$@"
execute
