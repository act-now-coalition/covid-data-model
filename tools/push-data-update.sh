#!/bin/bash

now=$(date)
git config --local user.email "action@github.com"
git config --local user.name "GitHub Action"
git add -A data
echo ${GITHUB_REF}
git commit --allow-empty -m "Updating saved datasets at $now"
git fetch
git push "https://${GITHUB_TOKEN}@github.com/${GITHUB_REPOSITORY}.git" HEAD:${GITHUB_REF}
