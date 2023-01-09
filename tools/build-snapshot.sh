#!/bin/bash
# build-snapshot.sh - Builds and publishes a new API snapshot (e.g.
# https://data.covidactnow.org/snapshot/123/).

set -o nounset
set -o errexit

CMD=$0

# Checks command-line arguments, sets variables, etc.
prepare () {
  # Parse args if specified.
  if [ $# -ne 1 ]; then
    exit_with_usage
  fi

  BRANCH=$1

  if [[ -z ${GITHUB_TOKEN:-} ]]; then
    echo "Error: GITHUB_TOKEN must be set to a personal access token. See:"
    echo "https://help.github.com/en/github/authenticating-to-github/creating-a-personal-access-token-for-the-command-line"
    exit 1
  fi

  # Set the sentry environment to allow quieting of alerts on
  # non-main branches.
  if [[ $BRANCH == "main" ]]; then
      SENTRY_ENVIRONMENT="production"
  else
      SENTRY_ENVIRONMENT="staging"
  fi
}

exit_with_usage () {
  echo "Usage: $CMD <branch (or commit sha, etc.)>"
  exit 1
}

execute () {
  curl -H "Authorization: token $GITHUB_TOKEN" \
       -H "Accept: application/vnd.github.v3+json" \
      --request POST \
      --data "{ \"ref\": \"${BRANCH}\", \"inputs\": { \"sentry_environment\": \"${SENTRY_ENVIRONMENT}\" } }" \
      https://api.github.com/repos/act-now-coalition/covid-data-model/actions/workflows/deploy_api.yml/dispatches

  echo "Publish requested. Go to https://github.com/act-now-coalition/covid-data-model/actions to monitor progress."
}

prepare "$@"
execute
