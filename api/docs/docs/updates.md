---
id: updates
title: API Updates
sidebar_label: API Updates
description: Updates to the Covid Act Now API.
---

Updates to the API will be reflected here.

### CDC Community Level data now available 
_Added on 2022-04-05_

The CDC Community Level metric as well as the subcomponents of the CDC Community Level metric are now available within the Covid Act Now API. 

[Learn more about how the CDC Community Level is measured](https://covidactnow.org/covid-risk-levels-metrics). 

Fields added:
* `communityLevels.cdcCommunityLevel`: The raw CDC Community Level metric. It serves as an exact mirror of the CDC’s published data, which is available for counties only and typically updates once a week.
* `communityLevels.canCommunityLevel`: The Covid Act Now team’s version of CDC Community Level metric. It uses the same methodology as the CDC community level, but it is available for states and metros in addition to counties. It also uses different data sources in some cases (New York Times for cases, HHS for state hospitalization data). It updates daily, though county hospitalization data only updates once a week.
* `metrics.weeklyNewCasesPer100k`: The number of new COVID cases per week per 100K population.
* `metrics.weeklyCovidAdmissionsPer100k`: The number of new COVID hospital admissions per week per 100K population. For counties this is calculated at the [Health Service Area level](https://apidocs.covidactnow.org/data-definitions#health-service-areas).
* `metrics.bedsWithCovidPatientsRatio`: The ratio of staffed inpatient beds that are occupied by COVID patients. For counties this is calculated at the [Health Service Area level](https://apidocs.covidactnow.org/data-definitions#health-service-areas).
* `actuals.hospitalBeds.weeklyCovidAdmissions`: The number of new COVID hospital admissions per week.
* `actuals.hsaHospitalBeds.weeklyCovidAdmissions`: The number of new COVID hospital admissions per week measured at the [Health Service Area level](https://apidocs.covidactnow.org/data-definitions#health-service-areas).
* `hsaName`: The name of the Health Service Area.
* `hsaPopulation`: The population of the Health Service Area.
* `hsaHospitalBeds`: This is a mirror of the existing hospitalBeds field, but measured at the [Health Service Area level](https://apidocs.covidactnow.org/data-definitions#health-service-areas).
* `hsaIcuBeds`: This is a mirror of the existing icuBeds field, but measured at the [Health Service Area level](https://apidocs.covidactnow.org/data-definitions#health-service-areas).


### Vaccine Booster data now available
_Added on 2022-01-13_

Vaccine booster shot (or additional dose) data is now available within the Covid Act Now API.

Fields added:
  * `actuals.vaccinationsAdditionalDose`: Number of individuals who are fully vaccinated and have received a booster (or additional) dose.
  * `metrics.vaccinationsAdditionalDoseRatio`: Ratio of population that are fully vaccinated and have received a booster (or additional) dose.

### ICU Headroom and Typical Usage Rate removed
_Added on 2021-09-16_

The following deprecated fields have been removed from the API:
`icuHeadroomRatio`, `icuHeadroomDetails`, and `typicalUsageRate`. Consider using
`icuCapacityRatio` instead.


### CDC Community Transmission Levels
_Added on 2021-07-30_

We now expose a CDC Community Transmission Level in the API.

The CDC community transmission levels are similar to the Covid Act Now risk levels, but have slightly different thresholds.
See [definitions of CDC community transmission levels](https://covid.cdc.gov/covid-data-tracker/#cases_community) for more details.

We calculate the level using the CDC's thresholds and expose it in the field
``cdcTransmissionLevel`` in all API responses.

The values correspond to the following levels:

| API value | CDC community transmission level |
| ------------ | ------------------------------- |
| 0 | Low |
| 1 | Moderate |
| 2 | Substantial |
| 3 | High |
| 4 | Unknown |

Note that the value may differ from what the CDC website reports given we have different data sources. We have also introduced an "Unknown" level for when both case data and test positivity data are missing for at least 15 days. The CDC does not have an "Unknown" level and instead will designate a location as "Low" when case and test positivity data are missing.


### Aggregated US data
_Added on 2021-03-31_

You can now query [US aggregated data](/#aggregated-us-data).

### Vaccine demographic data
_Added on 2021-03-25_

We are now starting to include vaccine demographic data; currently we are collecting two fields: `actuals.vaccinesAdministeredDemographics` and `actuals.vaccinationsInitiatedDemographics`. While we currently only have county level data in TX and PA (as of 3/25) we are working on adding more regions to provide the most complete vaccine demographic data.

Note that demographic buckets may differ by jurisdiction as different states may collect and bucket
demographic data in different ways. We surface the buckets as collected for now but this may change
in the future.

### Vaccines administered
_Added on 2021-03-23_

Added `actuals.vaccinesAdministered` to the API. This represents the total number of doses
administered for a region.

### New deaths column
_Added on 2021-03-08_

Added `actuals.newDeaths` and `actualsTimeseries.*.newDeaths` to the API.
The processing is similar to `actuals.newCases` - `newDeaths` represent new deaths
since previous report with erratic values removed by outlier detection.

### Field level annotations
_Added on 2021-01-19_

The Annotations field has a FieldAnnotations for each field in `Actuals`. You can now access the
data source(s) used to produce a field and list of dates where an anomalous observation was removed.
The exact structure of the `AnomalyAnnotation` may be modified in the future.

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
* `vaccinationsInitiatedRatio`: Ratio of population that has initiated vaccination.
* `vaccinationsCompletedRatio`: Ratio of population that has completed vaccination.

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
