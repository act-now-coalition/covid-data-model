# Covid Act Now API (V1)

## Introduction

The Covid Act Now API provides the same data that powers [CovidActNow.org](https://covidactnow.org) but in an easily digestible, machine readable format, intended for consumption by other COVID websites, models, and tools. Read our [blog post](https://blog.covidactnow.org/covid-act-now-api-intervention-model/) annoucing the API for more context and background.

### Update frequency

Data is updated every day, typically around midnight US Pacific Time.

### Rate Limits

There are no rate limits, but please be aware of your usage as we're a non-profit and would like to stay available to everyone.

### License

Data is licensed under [Creative Commons 4.0 By Attribution](https://creativecommons.org/licenses/by/4.0/). You are welcome to share, copy, and redistribute it, as well as adapt it for your own works, we just ask that you provide attribution to the source (as we have done with [our data sources](https://github.com/covid-projections/covid-data-public#date-sources-for-current--future-use)).


## Using the API

Data is available by prefixing URLs with `https://data.covidactnow.org/latest/`

In order to read a model from the API, you must specify both the location (state or county) and the intervention level. You can optionally specify if you'd like timeseries data.

### Location

Specify either:

1. A US State using two letter abbreviation (eg. 'CA' for California)
2. A US County using it's [FIPS Code](https://en.wikipedia.org/wiki/FIPS_county_code) (For a list of FIPS Codes, see [this page on the USDA site](https://www.nrcs.usda.gov/wps/portal/nrcs/detail/national/home/?cid=nrcs143_013697))
3. `/states/` or `/counties/` to get aggregate projections for all states or counties

### Invervention Categories

Forward projections are available for the following scenarios:

```js
"NO_INTERVENTION",          // No Intervention
"WEAK_INTERVENTION"         // Social Distancing
"STRONG_INTERVENTION"       // Stay at Home
"OBSERVED_INTERVENTION"     // Dynamic forecast based on observations
```


To get a dynamic forecast that is based on the actually observed effect of mitigations and other factors in a given state, use:

```js
"OBSERVED_INTERVENTION"
```

> Note: `OBSERVED_INTERVENTION` is only available for states, not counties.

More information on interventions, including definitions, references, and R0 values used is [available here](https://data.covidactnow.org/Covid_Act_Now_Model_References_and_Assumptions.pdf).

### Projected Overloads vs Timeseries

An optional parameter `timeseries` can be added before the file format, for example: `*.timeseries.json` or `*.timeseries.csv`.

If omitted, the API will return the date of projected hospital overloads, data of peak hospitalizations, and more.

If included, the API will return the projected hospitalization data every day for the next 90 days.

## Using the API
### Fetching State Data
#### Projections for a Specific State

Returns projections for the selected state

```bash
# Current actuals + projections + limits
# e.g. https://data.covidactnow.org/latest/us/states/CA.OBSERVED_INTERVENTION.json
/us/states/<ST>.<INTERVENTION>.json

# Full timeseries data: actuals + projected limits + data for every four days
# e.g. https://data.covidactnow.org/latest/us/states/CA.OBSERVED_INTERVENTION.timeseries.json
/us/states/<ST>.<INTERVENTION>.timeseries.json
```

#### Aggregate Projections for All States

Returns projections for all states

```bash
# Current actuals + projections + limits
# e.g. https://data.covidactnow.org/latest/us/states.OBSERVED_INTERVENTION.json
/us/states.<INTERVENTION>.json

# Timeseries data
# e.g. https://data.covidactnow.org/latest/us/states.OBSERVED_INTERVENTION.timeseries.json
/us/states.<INTERVENTION>.timeseries.json
```

State aggregates are also available as CSV files:

```bash
# Current actuals + projections + limits
# e.g. https://data.covidactnow.org/latest/us/states.OBSERVED_INTERVENTION.csv
/latest/us/states.<INTERVENTION>.csv

# Timeseries data
# E.G. https://data.covidactnow.org/latest/us/states.OBSERVED_INTERVENTION.timeseries.csv
/latest/us/states.<INTERVENTION>.timeseries.csv
```

### Fetching County Data
#### Projections for a Specific County

Returns projections for the selected county

```bash
# Current actuals + projections + limits
# e.g. https://data.covidactnow.org/latest/us/counties/06077.WEAK_INTERVENTION.json
/us/counties/<5-DIGIT-FIPS>.<INTERVENTION>.json

# Full timeseries data: actuals + projected limits + data for every four days
# e.g. https://data.covidactnow.org/latest/us/counties/06077.WEAK_INTERVENTION.timeseries.json
/latest/us/counties/<5-DIGIT-FIPS>.<INTERVENTION>.timeseries.json
```

#### Aggregate Projections for All Counties

Returns projections for all counties

```bash
# Current actuals + projections + limits
# e.g. https://data.covidactnow.org/latest/us/counties.WEAK_INTERVENTION.json
/us/counties.<INTERVENTION>.json

# Timeseries data
# e.g. https://data.covidactnow.org/latest/us/counties.WEAK_INTERVENTION.timeseries.json
/us/counties.<INTERVENTION>.timeseries.json
```

County aggregates are also available as CSV files:

```bash
# Current actuals + projections + limits
# e.g. https://data.covidactnow.org/latest/us/counties.WEAK_INTERVENTION.csv
/latest/us/counties.<INTERVENTION>.csv

# Timeseries data
# e.g. https://data.covidactnow.org/latest/us/counties.WEAK_INTERVENTION.timeseries.csv
/latest/us/counties.<INTERVENTION>.timeseries.csv
```

### Data format:

This is the data format for both states and counties. `timeseries` and `actualsTimeseries`
are only included when requesting `*.timeseries.json` or `*.timeseries.csv`.
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
    intervention, // one of (NO_INTERVENTION, WEAK_INTERVENTION, STRONG_INTERVENTION, OBSERVED_INTERVENTION)
    cumulativePositiveTests,
    cumulativeNegativeTests,
    cumulativeConfirmedCases,
    cumulativeDeaths,
    hospitalBeds: {
      capacity,  // *deprecated*
      totalCapacity,
      currentUsageCovid,
      currentUsageTotal,
      typicalUsageRate
    },
    ICUBeds: { same as above },  // Coming soon where available, null currently
    contactTracers
  },
  projections: {
    totalHospitalBeds: {
      shortageStartDate, // null if no shortage projected
      peakDate,
      peakShortfall
    },
    ICUBeds: { same as above },
    Rt,
    RtCI90
  },
  timeseries: [{
    date,
    hospitalBedsRequired,
    hospitalBedCapacity,
    ICUBedsInUse,
    ICUBedCapacity,
    ventilatorsInUse,
    RtIndictator,
    RtIndicatorCI90,
    cumulativeDeaths,
    cumulativeInfected,
    currentInfected,
    currentSusceptible,
    currentExposed
  }],
  actualsTimeseries: [{
    date,
    population,
    intervention, // one of (NO_INTERVENTION, WEAK_INTERVENTION, STRONG_INTERVENTION, OBSERVED_INTERVENTION)
    cumulativePositiveTests,
    cumulativeNegativeTests,
    cumulativeConfirmedCases,
    cumulativeDeaths,
    hospitalBeds: {
      capacity,  // *deprecated*
      totalCapacity,
      currentUsageCovid,
      currentUsageTotal,
      typicalUsageRate
    },
    ICUBeds: { same as above },  // Coming soon where available, null currently
    contactTracers
  }]
};
```
## Breaking Changes
* As of 6/5, `cumulativePositiveTests` and `cumulativeNegativeTests` were removed from the `timeseries` rows.  This data is still available in the `actualsTimeseries` field.
