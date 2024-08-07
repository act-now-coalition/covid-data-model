#!/bin/bash
# deploy-api-infrastructure.sh - Deploys API infrastructure to cloudfront lambda function.

set -o nounset
set -o errexit

# Checks command-line arguments, sets variables, etc.
prepare () {
  # Parse args if specified.
  if [ $# -ne 1 ]; then
    exit_with_usage
  else
    ENV=$1
  fi

  if [[ -z ${AWS_ACCESS_KEY_ID:-} || -z ${AWS_SECRET_ACCESS_KEY:-} || -z ${CLOUDFRONT_DISTRIBUTION_ID:-} || -z ${DOTENV:-} ]]; then
    echo "Error: AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY must be set to valid S3 credentials and CLOUDFRONT_DISTRIBUTION_ID and DOTENV must be set."
    exit 1
  fi
}

exit_with_usage () {
  echo "Usage: $CMD <env> <dotenv-path>"
  exit 1
}



execute() {
  cat > .env << EOF
$DOTENV
EOF

  # Run test to make sure that config properly loads - do not want to deploy code if config file
  # does not work.
  pytest unit_test.py::test_aws_auth_config_loads

  sls config credentials \
      --provider aws \
      --key $AWS_ACCESS_KEY_ID \
      --secret AWS_SECRET_ACCESS_KEY

  # Install python requirements and pare python libraries to fit Cloudfront lambda
  # 1MB max package size
  sls requirements install
  find .requirements | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf
  rm .requirements/**/*.so
  sls deploy -s $ENV

  aws cloudfront wait distribution-deployed --id $CLOUDFRONT_DISTRIBUTION_ID

  # Make sure API endpoints are working after deploy. Pulls from existing user and makes
  # sure that API returns results.
  pytest end_to_end_test.py::test_api_flow_existing_user
}

prepare "$@"
execute
