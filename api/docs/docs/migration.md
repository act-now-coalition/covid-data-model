---
id: migration
title: Covid Tracking Migration Guide
description: Migrating from the Covid Tracking Project
---

On March 7th, [The Covid Tracking Project](https://covidtracking.com) (CTP) will be winding down their daily updates. Throughout the pandemic they have provided an amazing service and resource with their daily data collection and in-depth reporting. For those looking for a replacement, the Covid Act Now API can be used to serve many of the same use cases.

The Covid Act Now API provides access to comprehensive COVID data — both current and historical. The data is available for all US states, counties, and metros and is aggregated from a number of official sources, quality assured, and updated daily.

Our API powers all data available on [covidactnow.org](https://covidactnow.org) including daily updates to cases, hospitalization, testing, and vaccination data. It includes raw data, risk levels, and calculated metrics to help you understand covid spread for a location. In addition to state data, we also provide county and metro data where available.

## Covid Act Now API Overview

In general, the Covid Act Now API provides much of the same data as The Covid Tracking Project. However there are some differences:

**County and metro data**  
In addition to state-level data as was provided by Covid Tracking Project, we also provide county and metro data where available. Typically county data is not as complete as state data but coverage is improving. Our county data is collected from a wide variety of sources that include federal, state, and local dashboards.

M**etrics and risk levels**  
We calculate metrics and risk levels derived from the raw data. These metrics include daily new cases per 100k, infection rate (R_t), test positivity, percent of population vaccinated, and ICU utilization.

**Vaccination data**  
Our API includes vaccination data. We include doses distributed, vaccinations initiated, and vaccinations completed.

**Testing data**  
For testing data, we focus on test positivity via PCR specimens which has become the standard metric tracked by most health departments. We have positive and negative PCR tests for all states and a computed test positivity percentage for all states and most counties.

**Hospitalization data**  
We currently ingest hospitalization data at the state and county level from HHS. For both ICU and overall hospitalization we include total staffed beds, beds in use by COVID patients, and total beds in use.

Many of the fields in the Covid Tracking API do overlap. Cases, deaths, and hospitalization/ICU data have the largest commonalities.

| CTP field name        | CAN Field Name                         |
| --------------------- | -------------------------------------- |
| date                  | date                                   |
| fips                  | fips                                   |
| state                 | state                                  |
| death                 | actuals.deaths                         |
| hospitalizedCurrently | actuals.hospitalBeds.currentUsageCovid |
| inIcuCurrently        | actuals.icuBeds.currentUsageCovid      |
| negativeTestsViral    | actuals.negativeTests                  |
| positiveTestsViral    | actuals.positiveTests                  |
| positive              | actuals.cases                          |
| positiveIncrease      | actuals.newCases                       |

## **Getting Started**

To get started, [register to get your API key](/access).

All requests should be made to `https://api.covidactnow.org` and include an API key. So for example, to query a CSV for the current values for all states, the request would be of the form:

    https://api.covidactnow.org/v2/states.csv?apiKey=YOUR_API_KEY

Many of endpoints from The Covid Tracking Project are available in our API. Below is a summary of similar endpoints:

| Description                          | Covid Tracking endpoint          | Covid Act Now endpoint             |
| ------------------------------------ | -------------------------------- | ---------------------------------- |
| Current values for all states        | /v1/states/current.{csv,json}    | /v2/states.{csv,json}              |
| Historical values for all states     | /v1/states/daily.{csv,json}      | /v2/states.timeseries.{csv,json}   |
| Current values for a single state    | /v1/states/ca/daily.{csv,json}   | /v2/state/CA.timeseries.{csv,json} |
| Historical values for a single state | /v1/states/ca/current.{csv,json} | /v2/state/CA.{json}                |
