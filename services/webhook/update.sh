#!/bin/sh

git pull origin main
git lfs prune
make -C services/data-pipeline-dashboard restart
date >> services/webhook/update-times.log
