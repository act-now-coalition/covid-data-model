# API

## Current Schema

* **/**
  * **version.json** - Metadata about how the API artifacts were generated.
    * *timestamp* (string) - an ISO 8601-formatted UTC timestamp.
    * *covid-data-public*
      * *branch* (string) - Branch name (usually "master").
      * *hash* (string) - Commit hash that branch was synced to.
      * *dirty* (boolean) - Whether there were any uncommitted / untracked files
        in the repo (usually false).
    * *covid-data-model*
      * *branch* (string) - Branch name (usually "master").
      * *hash* (string) - Commit hash that branch was synced to.
      * *dirty* (boolean) - Whether there were any uncommitted / untracked
        files in the repo (usually false).

We currently generate 4 types of files:
* arrays of arrays
* csv's
* shapefiles
* json summary files
