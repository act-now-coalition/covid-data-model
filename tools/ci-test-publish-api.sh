#!/bin/bash
# push-api.sh - Builds and publishes a new API snapshot (e.g.
# https://data.covidactnow.org/snapshot/123/).

set -o nounset
set -o errexit

CMD=$0

# Checks command-line arguments, sets variables, etc.
prepare () {
  # Parse args if specified.
  if [ $# -ne 0 ]; then
    exit_with_usage
  fi

  if [[ -z ${GITHUB_TOKEN:-} ]]; then
    echo "Error: GITHUB_TOKEN must be set to a personal access token. See:"
    echo "https://help.github.com/en/github/authenticating-to-github/creating-a-personal-access-token-for-the-command-line"
    exit 1
  fi
}

exit_with_usage () {
  echo "Usage: $CMD"
  exit 1
}

execute () {
  curl -H "Authorization: token $GITHUB_TOKEN" \
      --request POST \
      --data "{\"event_type\": \"ci-test-publish-api\" }" \
      https://api.github.com/repos/act-now-coalition/covid-data-model/dispatches

  echo "Publish requested. Go to https://github.com/act-now-coalition/covid-data-model/actions to monitor progress."
}

prepare "$@"
execute
