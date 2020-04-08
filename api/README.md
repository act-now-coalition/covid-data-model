# API

## Current Schema

We currently generate 4 file formats:
* arrays of arrays
* csv's
* shapefiles
* json summary files


### Current Structure
(pre-launch of v0)

```
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