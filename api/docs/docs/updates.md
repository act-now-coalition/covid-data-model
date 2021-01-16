---
id: updates
title: API Updates
sidebar_label: API Updates
description: Updates to the Covid Act Now API.
---

Updates to the API will be reflected here.

### Vaccine data now available
_Added on 2021-01-14_

Vaccine data is now available within the Covid Act Now API.

Currently the data is available for states only, but county-level vaccination data is coming soon.

Fields added:
 * `vaccinesDistributed`: Total number of vaccine doses distributed.
 * `vaccinationsInitiated`: Total number of people initiating vaccination. For a vaccine with a
   2-dose regimen, this represents the first dose.
 * `vaccinationsCompleted`: Total number of people completing vaccination - currently those
    completing their second shot. 

You can access these fields in both the `actuals` field and `actualsTimeseries` fields.

### View entire timeseries of risk levels for all regions
_Added on 2020-12-22_

You can now view the history of a region's overall risk level in all timeseries endpoints under the
key `riskLevelsTimeseries`.

### Overall risk level now based on 3 key metrics
_Added on 2020-12-22_

The overall risk level is now based on `caseDensity`, `testPositivityRatio`, and `infectionRate`.
Learn more about the [changes we made](https://covidactnow.org/faq#december-risk-levels-change).

We will be continuing to calculate all metrics and have no plans of removing
`contactTracingCapacityRatio`, `icuCapacityRatio`, or `icuHeadroomRatio` at this time.

### Link to Covid Act Now website
_Added on 2020-12-03_

Each region now includes a field `url` that points to the Covid Act Now location page
for that region.

### Upcoming risk level update
_Added on 2020-12-01_

We modified our risk levels to include a 5th level on
[covidactnow.org](https://covidactnow.org) for locations experiencing a
severe outbreak. On **Monday, December 7th** we will be updating the risk levels
in the API to reflect this.

If you have any code depending on 4 risk levels, you will need to update
it to include the fifth risk level.

This change directly affects the fields `riskLevels.overall` and `riskLevels.caseDensity`.

```diff
RiskLevel:
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3
    UNKNOWN = 4
+   EXTREME = 5
```

If you would like to include the new risk level right now to reflect what
is currently on [covidactnow.org](https://covidactnow.org), you can do so by
classifying both `overall` and `caseDensity` risk as extreme for any location
where `actuals.casesDensity > 75`.

### Query by CBSA
_Added on 2020-11-09_

We added [core-based statistical area](https://en.wikipedia.org/wiki/Core-based_statistical_area)
endpoints to the API.

You can now view results aggregated by CBSA:
```bash
# Aggregate CBSA data
https://api.covidactnow.org/v2/cbsas.json?apiKey={apiKey}
https://api.covidactnow.org/v2/cbsas.timeseries.json?apiKey={apiKey}
https://api.covidactnow.org/v2/cbsas.csv?apiKey={apiKey}
https://api.covidactnow.org/v2/cbsas.timeseries.csv?apiKey={apiKey}

# Individual CBSAs
https://api.covidactnow.org/v2/cbsa/{cbsa_code}.json?apiKey={apiKey}
https://api.covidactnow.org/v2/cbsa/{cbsa_code}.timeseries.json?apiKey={apiKey}
```

Read the [CBSA API Documentation](/api#tag/CBSA-Data) to learn more.

### Increase county test positivity coverage
_Added on 2020-11-04_

We increased our test positivity coverage for counties by adding in data
from U.S. Department of Health & Human Services as aggregated by the
Centers for Medicare & Medicaid Services. See our [data sources](https://docs.google.com/presentation/d/1XmKCBWYZr9VQKFAdWh_D7pkpGGM_oR9cPjj-UrNdMJQ/edit#slide=id.ga721750846_35_117)
for more information.

### Add new cases field with outlier detection
_Added on 2020-10-29_

In addition to cumulative case counts, we added a `New Cases` field to all
`actuals` and `actualsTimeseries` values.  The `New Cases` field computes new
cases and applies outlier detection to remove erratic case values.  

### Add `locationId` field
_Added on 2020-10-27_

Adds a generic location ID used to represent regions.  Will allow for 
greater flexibility as we add more aggregation levels (such as country).

### Add `riskLevels`
_Added on 2020-10-15_

Added risk levels as seen on [covidactnow.org](https://covidactnow.org) to
the API.