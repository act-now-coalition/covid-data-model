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

### Arrays

For state `<state abreviation>.<intervention number>.json` files
and for county `<state abreviation>.<fips>.<intervention number>.json` files

we have an array of arrays as such `[items... ]` where each item represents:

```json
   [
        "day_index",
        "date",
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
        "all_hospitalized",
        "all_infected",
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

`counties.<intervention>.csv`
`states.<intervention>.csv`

Each csv has
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

Here's an example CSV
```csv
OBJECTID,Province/State,Country/Region,Last Update,Latitude,Longitude,State/County FIPS Code,State Intervention,16d-HSPTLZD,32d-HSPTLZD,16d-LACKBEDS,32d-LACKBEDS,MEAN-HOSP,MEAN-DEATHS,PEAK-HOSP,PEAK-DEATHS,Current Deaths,Current Confirmed,Combined Key,County
0,South Carolina,US,4/7/2020 23:04,34.22333378,-82.46170658,45001,shelter_in_place,3,5,0,0,6.0,0.3333333333333333,2020-06-19,2020-05-08,0,5,"Abbeville, South Carolina, US",Abbeville
1,South Carolina,US,4/7/2020 23:04,33.54338026,-81.63645384,45003,shelter_in_place,18,30,0,0,34.0,2.4814814814814814,2020-06-19,2020-06-23,1,28,"Aiken, South Carolina, US",Aiken
```

### Shapefiles

Each set of shapefiles, namely for counties
`counties.<intervention>.shp`
`counties.<intervention>.dbf`
`counties.<intervention>.shx`

and for states
`states.<intervention>.shp`
`states.<intervention>.dbf`
`states.<intervention>.shx`

contains the original shapefile and additional attributes:

Here's an example
```
    State_Inter	shelter_in_place
    16d-HSPTLZD	10
    32d-HSPTLZD	28
    16d-LACKBED	0
    32d-LACKBED	0
    MEAN-HOSP	74
    MEAN-DEATHS	4
    PEAK-HOSP	2020-07-03 00:00:00
    PEAK-DEATHS	2020-07-03 00:00:00
    Current_Dea	0
    Current_Con	10
```

### Case Summary

For `case_summary/<state abreviation>.summary.json` files see the [JSON Schema](schemas/case_summary.json)

### County Summaries

For `county_summaries/<state abreviation>.summary.json` files see the [JSON Schema](schemas/county_summaries.json)


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
