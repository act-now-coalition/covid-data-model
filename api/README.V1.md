# Covid Act Now API (V1)

## Introduction

The Covid Act Now API provides the same data that powers [CovidActNow.org](https://covidactnow.org) but in an easily digestible, machine readable format, intended for consumption by other COVID websites, models, and tools. Read our [blog post](https://blog.covidactnow.org/covidactnow-api-launch/) annoucing the API for more context and background.

### Update frequency

Data is updated every day, typically around midnight US Pacific Time.

### Rate Limits

There are no rate limits

### License

The data presented in the Covid Act Now API is licensed under [Creative Commons 4.0 By Attribution](https://creativecommons.org/licenses/by/4.0/). You are welcome to share, copy, and redistribute it, as well as adapt it for your own works, we just ask that you provide attribution to the source (as we have done with [our data sources](https://github.com/covid-projections/covid-data-public#date-sources-for-current--future-use)).


## Using the API

Data is available by prefixing URLs with `https://data.covidactnow.org/latest/`

In order to read a model from the API, you must specify both the location (state or county) and the intervention level.

### Location

Specify either:

1. A US State using two letter abbreviation (eg. 'CA' for California)
2. A US County using it's [FIPS Code](https://en.wikipedia.org/wiki/FIPS_county_code) (For a list of FIPS Codes, see [this page on the USDA site](https://www.nrcs.usda.gov/wps/portal/nrcs/detail/national/home/?cid=nrcs143_013697))

### Invervention Categories

Forward projections are available for the following scenarios:

```js
"NO_MITIGATION",          // No Intervention
"MODERATE_MITIGATION"     // Social Distancing
"HIGH_MITIGATION"         // Stay at Home
```

Additionally the most appropriate static scenario based on the per-state intervention is returned by specifying:

```js
"SELECTED_MITIGATION"
```

To get a dynamic forecast that is based on the actually observed effect of mitigations and other factors in a given state, use:

```js
"OBSERVED_MITIGATION"
```

> Note: `OBSERVED_MITIGATION` is only available for states, not counties.

More information on interventions, including definitions, references, and R0 values used is [available here](https://data.covidactnow.org/Covid_Act_Now_Model_References_and_Assumptions.pdf).

### Fetching State Data
#### Reading a Projection for a Specific State

Returns projections for the selected state

```bash
# Current actuals + projections + limits
# e.g. https://data.covidactnow.org/latest/us/states/CA.OBSERVED_MITIGATION.json
/us/states/<ST>.<INTERVENTION>.json

# Full timeseries data: actuals + projected limits + data for every four days
# e.g. https://data.covidactnow.org/latest/us/states/CA.OBSERVED_MITIGATION.timeseries.json 
/us/states/<ST>.<INTERVENTION>.timeseries.json
```

#### Reading Aggregate Projections for All States

Returns projections for all states

```bash
# Current actuals + projections + limits
# e.g. https://data.covidactnow.org/latest/us/states.OBSERVED_MITIGATION.json
/us/states.<INTERVENTION>.json

# Timeseries data
# e.g. https://data.covidactnow.org/latest/us/states.OBSERVED_MITIGATION.timeseries.json
/us/states.<INTERVENTION>.timeseries.json
```

State aggregates are also available as CSV files:
    
```bash
# Current actuals + projections + limits
# e.g. https://data.covidactnow.org/latest/us/states.OBSERVED_MITIGATION.csv
/latest/us/states.<INTERVENTION>.csv

# Timeseries data
# E.G. https://data.covidactnow.org/latest/us/states.OBSERVED_MITIGATION.timeseries.csv
/latest/us/states.<INTERVENTION>.timeseries.csv
```

### Fetching County Data
#### Reading a Projection for a Specific County

Returns projections for the selected county
    
```bash
# Current actuals + projections + limits
# e.g. https://data.covidactnow.org/latest/us/counties/06077.SELECTED_MITIGATION.json
/us/counties/<5-DIGIT-FIPS>.<INTERVENTION>.json 

# Full timeseries data: actuals + projected limits + data for every four days
# e.g. https://data.covidactnow.org/latest/us/counties/06077.SELECTED_MITIGATION.timeseries.json
/latest/us/counties/<5-DIGIT-FIPS>.<INTERVENTION>.timeseries.json 
```

#### Reading Aggregate Projections for All Counties

Returns projections for all counties

```bash
# Current actuals + projections + limits
# e.g. https://data.covidactnow.org/latest/us/counties.SELECTED_MITIGATION.json
/us/counties.<INTERVENTION>.json

# Timeseries data
# e.g. https://data.covidactnow.org/latest/us/counties.SELECTED_MITIGATION.timeseries.json
/us/counties.<INTERVENTION>.timeseries.json
```

County aggregates are also available as CSV files:
    
```bash
# Current actuals + projections + limits
# e.g. https://data.covidactnow.org/latest/us/counties.SELECTED_MITIGATION.csv
/latest/us/counties.<INTERVENTION>.csv

# Timeseries data
# e.g. https://data.covidactnow.org/latest/us/counties.SELECTED_MITIGATION.timeseries.csv
/latest/us/counties.<INTERVENTION>.timeseries.csv
```

### Data format:

This is the data format for both states and counties. `timeseries` is only included when requesting `*.timeseries.json` or `*.timeseries.csv`.
```jsonc
{
  country,
  stateName,
  countyName, // null for states
  fips, // 2 digit for states, 5 digit for counties
  lat, 
  long,
  lastUpdatedDate, // ISO 8601 date string
  actuals: {
    population,
    intervention, // one of (NO_MITIGATION, MODERATE_MITIGATION, stay_at_home)
    cumulativeConfirmedCases,
    cumulativeDeaths,
    hospitalBeds: {
      capacity,
      currentUsage, // Coming soon where available, null currently
    }, 
    ICUBeds: { same as above }  // Coming soon where available, null currently
  }, 
  projections: {
    totalHospitalBeds: {
      shortageStartDate, // null if no shortage projected
      peakDate,
      peakShortfall
    },
    ICUBeds: { same as above }, // Coming soon where available, null currently
  },
  timeseries: [{
    date,
    hospitalBedsRequired,
    hospitalBedCapacity,
    ICUBedsInUse,
    ICUBedCapacity, // Coming soon where availabe, null currently
    cumulativeDeaths,
    cumulativeInfected,
  }],
};
```
## Coming soon
* Hospital bed usage (actuals)
* ICU bed data (capacity, projections, and actuals)
* More file forms (dbf,shp,shx)

