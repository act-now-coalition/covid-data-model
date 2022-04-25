# (DEPRECATED) V0 API

## :warning: API Deprecated

API V1 is now deprecated and will stop daily updates on *Monday, October 5th*.

Please refer to the [documentation](https://apidocs.covidactnow.org) for the latest version of the API.

For assistance migrating to V2, check out the [migration guide](https://apidocs.covidactnow.org/migration).

If you have any questions, do not hesitate to reach out to <api@covidactnow.org>. Thanks!

## V0 Schema Overview

**This API schema is deprecated and we recommend the v1 API schema linked above**

We currently provide 6 file types:
* Arrays of Arrays, by State & County
* CSV's, by State & County
* Shapefiles, by State & County
* Case JSON Summary
* County JSON Summaries
* Versions.json

### Current Structure

```bash
├── /latest/ | /snapshots/<snapshot id>/
│   └── case_summary/
│   │   ├── <state abreviation>.summary.json
│   │   └── case_summary.version.json
│   ├── county/
│   │   ├── <state abreviation>.<fips>.<intervention number>.json
│   │   └── county.version.json
│   ├── us/
|   │   ├── counties/
|   │   │   ├── <5 Digit FIPS>.<intervention number>.json
|   │   │   └── counties_top_100.json
|   │   └── states/
|   │       └── <state abbreviation>.<intervention number>.json
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


## Generating a new API schema

To generate a new tracked API schema, create a new python file in the `api/` folder containing
a `pydantic.BaseModel` class definition. Read [these docs](https://pydantic-docs.helpmanual.io/usage/schema/)
to learn more about how pydantic generates json-schema from python objects.

To update the public API, run:
```
./run.py api update-schemas
```

This will find all python classes under `api/` that subclass `pydantic.BaseModel` and
generate the corresponding JSON Schema file into `api/schemas/`.

If you do not want to generate a top-level schema in `api/schemas/`, prepend the class name
with a `_` (i.e. `_MyPrivateSchema`); these will not be uploaded.


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

For `case_summary/<state abreviation>.summary.json` files see the [JSON Schema](schemas/StateCaseSummary.json)

### County Summaries

For `county_summaries/<state abreviation>.summary.json` files see the [JSON Schema](schemas/county_summaries.json)

### Top Counties
For `county/counties_top_100.json` fields see the [JSON Schema](schema/CANPredictionAPI.json)

### For State/County Calculated Results
For files like
- `us/counties/<fips>.<intervention>.json`
- `us/states/<state abbreviation>.<intervention>.json`

see the [JSON Schema](schema/CANPredictionAPIRow.json)

### Versions

* **/**
  * **version.json** - Metadata about how the API artifacts were generated.
    * *timestamp* (string) - an ISO 8601-formatted UTC timestamp.
    * *covid-data-model*
      * *branch* (string) - Branch name (usually "main").
      * *hash* (string) - Commit hash that branch was synced to.
      * *dirty* (boolean) - Whether there were any uncommitted / untracked
        files in the repo (usually false).
