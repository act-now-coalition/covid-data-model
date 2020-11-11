---
id: updates
title: API Updates
sidebar_label: API Updates
description: Updates to the API.
---

Updates to the API will be reflected here.

### Query by CBSA
_Added on 2020-11-09_

We added core-based statistical area endpoints to the API.

Read the [CBSA API Documentation](/api#tag/CBSA-(Metro)-Data) to learn more.


### Increase county test positivity coverage
_Added on 2020-11-04_

We increased our test positivity coverage for counties.  
We now pull from CMS...

### Add new cases field with outlier detection
_Added on 2020-10-29_

In addition to cumulative case counts, we added a `New Cases` field to all
`actuals` and `actualsTimeseries` values.  The `New Cases` field computes new
cases and applies outlier detection to remove erratic case values.  

### Add `locationId` field
_Added on 2020-10-27_

Adds a generic location ID used to represent regions.  Will allow for 
greater flexibility as we add more aggregation levels (such as country).



