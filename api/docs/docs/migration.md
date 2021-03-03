---
id: migration
title: Migrating from The Covid Tracking Project
description: Migrating from the Covid Tracking Project
---


On March 7th, [The Covid Tracking Project](https://covidtracking.com) (CTP) will be winding down their daily updates. Throughout the pandemic they have provided an amazing service and resource with their daily data collection and in-depth reporting. We think the Covid Act Now API can be a helpful resource to those looking for similar data.

The Covid Act Now API provides access to comprehensive COVID data — both current and historical. The data is available for all US states, counties, and metros and is aggregated from a number of official sources, quality assured, and updated daily.

Our API powers all data available on [covidactnow.org](https://covidactnow.org) including daily updates to cases, hospitalization, testing, and vaccination data. It includes raw data, risk levels, and metrics (including daily new cases, infection rate, test positivity, and ICU utilization) to help you understand covid spread for a location.

We also provide county and metro data where available. Typically county data is not as complete as state data but coverage is improving. Our county data is collected from a wide variety of sources that include federal, state, and local dashboards.

If you’re a current CTP user, hopefully we are able to help fill your COVID data needs after March 7th.  

## Covid Act Now API Overview

In general, the Covid Act Now API provides much of the same data as Covid Tracking. However there are some differences:

**Our API includes slightly different hospitalization data**  
We include current bed capacity and total beds occupied as well (including COVID and non-COVID patients). We currently ingest hospitalization data at the state and county level from HHS.

**Vaccination data**  
Our API includes vaccination data. We include doses distributed, vaccinations initiated, and vaccinations completed.

**Additional metrics and risk levels**  
We calculate metrics and risk levels derived from the raw data.  These metrics include but are not limited to daily new cases per 100k, infection rate, and test positivity.

**CTP has much more detailed testing data**  
When CTP was started, it initially focused on testing data whereas we focused on broader datasets, other metrics, and county data. While not immediately planned, we may add more detailed testing data in the future. We have focused more on surfacing test positivity - and provide it for all states and most counties.

**Our API requires an API Key**  
Something

Many of the fields in the Covid Tracking API do overlap.  Cases, deaths, and hospitalization/ICU data have the largest commonalities. 

| CTP field name        | CAN Field Name                         |
| --------------------- | -------------------------------------- |
| date                  | date                                   |
| fips                  | fips                                   |
| state                 | state                                  |
| death                 | actuals.deaths                         |
| hospitalizedCurrently | actuals.hospitalBeds.currentUsageCovid |
| inIcuCurrently        | actuals.icuBeds.currentUsageCovid      |
| negative              | actuals.negativeTests                  |
| positive              | actuals.cases                          |
| positiveIncrease      | actuals.newCases                       |

As there are different sources for each of these values, the exact values may not match 1:1. 


## **Getting Started**

To get started, [register to get your API key](/access). 

Many of endpoints from The Covid Tracking Project are available in our API.  Below is a summary of similar endpoints:

| Description                          | Covid Tracking endpoint          | Covid Act Now endpoint             |
| ------------------------------------ | -------------------------------- | ---------------------------------- |
| Current values for all states        | /v1/states/current.{csv,json}    | /v2/states.{csv,json}              |
| Historical values for all states     | /v1/states/daily.{csv,json}      | /v2/states.timeseries.{csv,json}   |
| Current values for a single state    | /v1/states/ca/daily.{csv,json}   | /v2/state/CA.timeseries.{csv,json} |
| Historical values for a single state | /v1/states/ca/current.{csv,json} | /v2/state/CA.{csv,json}            |

Additional notes:

- All covid act now links must include the API key, so `/v2/states.json` would be `https://api.covdactnow.org/v2/states.json?apiKey=YOUR_API_KEY`. 
- When querying a specific state, states are required to be upper case. 

