# API

## Current Schema Overview

We currently provide 6 file types:
* Arrays of Arrays, by State & County
* CSV's, by State & County
* Shapefiles, by State & County
* Case JSON Summary
* County JSON Summaries
* Versions.json


### Current Structure
(pre-launch of v0)

```bash
├── /v0/<snapshot id>/
│   └── case_summary/
│   │   ├── <state abreviation>.summary.json
│   │   └── case_summary.version.json
│   ├── county/
│   │   ├── <state abreviation>.<fips>.<intervention number>.json
│   │   └── county.version.json
│   ├── county_summaries/
│   │   ├── <state abreviation>.summary.json
│   │   └── county_summary.version.json
│   ├── custom1/
│   │   ├── counties.<intervention>.csv
│   │   ├── states.<intervention>.csv
│   │   ├── counties.<intervention>.shp
│   │   ├── states.<intervention>.shp
│   │   ├── counties.<intervention>.dbf
│   │   ├── states.<intervention>.dbf
│   │   ├── counties.<intervention>.shx
│   │   ├── states.<intervention>.shx
│   ├── <state abreviation>.<intervention number>.json
│   └── states.version.json
```


## Specific Files Schemas

For a state <state abreviation>.<intervention number>.json
For a county <state abreviation>.<fips>.<intervention number>.json

### Arrays

[items... ]

Where each item is
```json
   [
        "day_index",
        "date",
        "a", # total
        "b", # susceptible
        "c", # exposed
        "d", # infected
        "e", # infected_a (not hospitalized, but infected)
        "f", # infected_b (hospitalized not in icu)
        "g", # infected_c (in icu)
        "all_hospitalized", # infected_b + infected_c
        "all_infected", # infected_a + infected_b + infected_c
        "dead",
        "beds",
        "i",
        "j",
        "k",
        "l",
        "population",
        "m",
        "n",
    ]
```

### CSVs

counties.<intervention>.csv
states.<intervention>.csv

```
    OBJECTID
    Province/State
    Country/Region
    Last Update
    Latitude
    Longitude
    State/County FIPS Code
    State Intervention
    16d-HSPTLZD
    32d-HSPTLZD
    16d-LACKBEDS
    32d-LACKBEDS
    MEAN-HOSP
    MEAN-DEATHS
    PEAK-HOSP
    PEAK-DEATHS
    Current Deaths
    Current Confirmed
    Combined Key
    County
```

Example CSV
```csv
OBJECTID,Province/State,Country/Region,Last Update,Latitude,Longitude,State/County FIPS Code,State Intervention,16d-HSPTLZD,32d-HSPTLZD,16d-LACKBEDS,32d-LACKBEDS,MEAN-HOSP,MEAN-DEATHS,PEAK-HOSP,PEAK-DEATHS,Current Deaths,Current Confirmed,Combined Key,County
0,South Carolina,US,4/7/2020 23:04,34.22333378,-82.46170658,45001,shelter_in_place,3,5,0,0,6.0,0.3333333333333333,2020-06-19,2020-05-08,0,5,"Abbeville, South Carolina, US",Abbeville
1,South Carolina,US,4/7/2020 23:04,33.54338026,-81.63645384,45003,shelter_in_place,18,30,0,0,34.0,2.4814814814814814,2020-06-19,2020-06-23,1,28,"Aiken, South Carolina, US",Aiken
```

### Shapefiles

counties.<intervention>.shp
states.<intervention>.shp
counties.<intervention>.dbf
states.<intervention>.dbf
counties.<intervention>.shx
states.<intervention>.shx

TODO:

### Case Summary

case_summary/<state abreviation>.summary.json

See the [JSON Schema](api/schemas/case_summary.json)

### County Summaries

county_summaries/<state abreviation>.summary.json

See the [JSON Schema](api/schemas/county_summaries.json)


### Versions

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