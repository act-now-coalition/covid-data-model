#!/usr/bin/env bash

git pull origin main
make -C services/data-pipeline-dashboard restart
date >> services/webhook/update-times.log
