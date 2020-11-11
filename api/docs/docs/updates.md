---
id: updates
title: API Updates
sidebar_label: API Updates
description: Updates to the Covid Act Now API.
---

Updates to the API will be reflected here.

### Query by CBSA
_Added on 2020-11-09_

We added [core-based statistical area](https://en.wikipedia.org/wiki/Core-based_statistical_area)
endpoints to the API.

You can now view results aggregated by CBSA:
```bash
# Aggregate CBSA data
https://api.covidactnow.org/latest/v2/cbsas.json?apiKey={apiKey}
https://api.covidactnow.org/latest/v2/cbsas.timeseries.json?apiKey={apiKey}
https://api.covidactnow.org/latest/v2/cbsas.csv?apiKey={apiKey}
https://api.covidactnow.org/latest/v2/cbsas.timeseries.csv?apiKey={apiKey}

# Individual CBSAs
https://api.covidactnow.org/latest/v2/cbsa/{cbsa_code}.json?apiKey={apiKey}
https://api.covidactnow.org/latest/v2/cbsa/{cbsa_code}.timeseries.json?apiKey={apiKey}
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