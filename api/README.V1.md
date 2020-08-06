# Covid Act Now API (V1)

## Introduction

The Covid Act Now API provides the same data that powers [CovidActNow.org](https://covidactnow.org) but in an easily digestible, machine readable format, intended for consumption by other COVID websites, models, and tools. Read our [blog post](https://blog.covidactnow.org/covid-act-now-api-intervention-model/) annoucing the API for more context and background.

### Update frequency

Data is updated every day, typically around midnight US Pacific Time.

## Notable Changes
* 7/16 - ``projections.Rt`` and ``projections.RtCI90`` now match the Rt in `timeseries.RtIndicator` and `timeseries.RtIndicatorCI90`.  This value now matches the Infection Rate shown on the website as opposed to a separate value inferred from our SEIR model.
* 6/5 - `cumulativePositiveTests` and `cumulativeNegativeTests` were removed from the `timeseries` rows.  This data is still available in the `actualsTimeseries` field.

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

Returns projections for the selected state:

## API Endpoints

| Endpoint | Description | Schema |
| -------- | ----------- | ------ |
| /us/states/<STATE>.<INTERVENTION>.json | State summary for <INTERVENTION> | RegionSummary |
| /us/states/<STATE>.<INTERVENTION>.timeseries.json | State timeseries for intervention | RegionSummaryWithTimeseries |
| /us/counties/<FIPS>.<INTERVENTION>.json | County summary for <INTERVENTION> | RegionSummary |
| /us/counties/<FIPS>.<INTERVENTION>.timeseries.json | County timeseries for <INTERVENTION> | RegionSummaryWithTimeseries |
| /us/states.<INTERVENTION>.{json,csv} | Summary for all states | AggregateRegionSummary |
| /us/states.<INTERVENTION>.timeseries.json | Timeseries data for all states | AggregateRegionSummaryWithTimeseries |
| /us/states.<INTERVENTION>.timeseries.csv | Timeseries data for all states | AggregateFlattenedTimeseries |
| /us/counties.<INTERVENTION>.{json,csv} | Summary for all counties | AggregateRegionSummary |
| /us/counties.<INTERVENTION>.timeseries.json | Timeseries data for all counties | AggregateRegionSummaryWithTimeseries |
| /us/counties.<INTERVENTION>.timeseries.csv | Timeseries data for all counties | AggregateFlattenedTimeseries |
