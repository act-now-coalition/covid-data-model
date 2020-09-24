---
id: migration
title: Migrating to v2
---


Lots has changed in the COVID data landscape since we create the first version of our API.
Our API largely was focused around interventions taken at the state level and the API structure reflected that.

Additionally, the data reported and collected has evolved. When we first launched our API few states were reporting
hospitalizations and even fewer reporting test data. 

:::note

API v1 is now deprecated.  We will continue to update v1 endpoints
until October 5th, 2020 and will remove access to v1 data on October 26th, 2020.

If you have any questions, please don't hesitate to reach out at <api@covidactnow.org>.

:::


### Top level metrics included

We are now surfacing the top level metrics that power the CAN risk levels.

These are available under `metrics` and `metricsTimeseries` keys.

The following metrics are available:
 - **caseDensity** - The number of cases per 100k population calculated using a 7-day rolling average.
 - **testPositivity** - The ratio of people who test positive calculated using a 7-day rolling average.
 - **infectionRate** - R<sub>t</sub>, or the estimated number of infections arising from a typical case.
 - **icuHeadroomRatio** - Remaining ICU headroom capacity for COVID patients.
 - **contactTracer** - Ratio of currently hired tracers to estimated tracers needed based on 7-day daily case average.


### All queries require an API key

We are now requiring an API key to access the data. All requests should include an
`apiKey` query parameter to authenticate.

Sign up [here](/access) to get an API key.

```diff
- https://data.covidactnow.org/latest/states.json
+ https://api.covidactnow.org/v2/states.json?apiKey={apiKey}
```

### Interventions are no longer included at the route level
We no longer surface projections by intervention.

Here are examples of API endpoint changes:
```diff
- https://data.covidactnow.org/latest/state/{state}.{intervention}.json
+ https://api.covidactnow.org/v2/state/{state}.json?apiKey={apiKey}
```
```diff
- https://data.covidactnow.org/latest/county/{fips}.{intervention}.json
+ https://api.covidactnow.org/v2/county/{fips}.json?apiKey={apiKey}
```
```diff
- https://data.covidactnow.org/latest/counties.{intervention}.json
+ https://api.covidactnow.org/v2/counties.json?apiKey={apiKey}
```
```diff
- https://data.covidactnow.org/latest/states.{intervention}.json
+ https://api.covidactnow.org/v2/states.json?apiKey={apiKey}
```

### Projections deprecated

The projections and projections timeseries are not included in the new API.
If you still need access to our intervention projection models, please use the [V1 API](https://github.com/covid-projections/covid-data-model/blob/master/api/README.V1.md). 

As our projection models evolve, we are planning on adding projections back in, but they may look
different than the old format.

### Fields renamed

Many fields have been slightly renamed for clarity.  Double check the [API Reference](/api) for details.

### States reported by state code
States are now reported by the 2-letter state code rather than the full state name.