#!/bin/bash

# Do a fresh pull in case any changes have been pushed to the branch.
git pull "https://${GITHUB_TOKEN}@github.com/${GITHUB_REPOSITORY}.git" HEAD:${GITHUB_REF}

now=$(date)
git config --local user.email "action@github.com"
git config --local user.name "GitHub Action"
git add -A data
git commit -m "Updating saved datasets at $now"

# test branch name
echo $GITHUB_REF

GIT_TRACE=1 GIT_CURL_VERBOSE=1 git push "https://${GITHUB_TOKEN}@github.com/${GITHUB_REPOSITORY}.git" HEAD:${GITHUB_REF}
