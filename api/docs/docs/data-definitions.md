---
id: data-definitions
title: Data Definitions
---

Read more about the data included in the Covid Act Now API.


## Cases

### Cases

  Cumulative confirmed or suspected cases.

**Where to access**  
* CSV column names: ``actuals.cases``
* JSON file fields: ``actuals.cases``, ``actualsTimeseries.*.cases``

### New Cases

  New confirmed or suspected cases.


New cases are a processed timeseries of cases - summing new cases may not equal
the cumulative case count.

Processing steps:
 1. If a region does not report cases for a period of time but then begins reporting again,
    we will exclude the first day that reporting recommences. This first day likely includes
    multiple days worth of cases and can be misleading to the overall series.
 2. We remove any days with negative new cases.
 3. We apply an outlier detection filter to the timeseries, which removes any data
    points that seem improbable given recent numbers. Many times this is due to
    backfill of previously unreported cases.

**Where to access**  
* CSV column names: ``actuals.newCases``
* JSON file fields: ``actuals.newCases``, ``actualsTimeseries.*.newCases``

### Case Density

  The number of cases per 100k population calculated using a 7-day rolling average.

**Where to access**  
* CSV column names: ``metrics.caseDensity``
* JSON file fields: ``metrics.caseDensity``, ``metricsTimeseries.*.caseDensity``

### Weekly New Cases Per 100k Population

  The number of new cases per 100k population over the last week.

**Where to access**  
* CSV column names: ``metrics.weeklyNewCasesPer100k``
* JSON file fields: ``metrics.weeklyNewCasesPer100k``, ``metricsTimeseries.*.weeklyNewCasesPer100k``

### Infection Rate

  R_t, or the estimated number of infections arising from a typical case.

**Where to access**  
* CSV column names: ``metrics.infectionRate``
* JSON file fields: ``metrics.infectionRate``, ``metricsTimeseries.*.infectionRate``

### Infection Rate Ci90

  90th percentile confidence interval upper endpoint of the infection rate.

**Where to access**  
* CSV column names: ``metrics.infectionRateCI90``
* JSON file fields: ``metrics.infectionRateCI90``, ``metricsTimeseries.*.infectionRateCI90``



## Tests

### Positive Tests

  Cumulative positive test results to date

**Where to access**  
* CSV column names: ``actuals.positiveTests``
* JSON file fields: ``actuals.positiveTests``, ``actualsTimeseries.*.positiveTests``

### Negative Tests

  Cumulative negative test results to date

**Where to access**  
* CSV column names: ``actuals.negativeTests``
* JSON file fields: ``actuals.negativeTests``, ``actualsTimeseries.*.negativeTests``

### Test Positivity Ratio

  Ratio of people who test positive calculated using a 7-day rolling average.

**Where to access**  
* CSV column names: ``metrics.testPositivityRatio``
* JSON file fields: ``metrics.testPositivityRatio``, ``metricsTimeseries.*.testPositivityRatio``

### Test Positivity Ratio Details

  

**Where to access**  
* CSV column names: ``metrics.testPositivityRatioDetails``
* JSON file fields: ``metrics.testPositivityRatioDetails``, ``metricsTimeseries.*.testPositivityRatioDetails``



## Hospitalizations

### Icu Beds

  Information about ICU bed utilization details.

Fields:
 * capacity - Current staffed ICU bed capacity.
 * currentUsageTotal - Total number of ICU beds currently in use
 * currentUsageCovid - Number of ICU beds currently in use by COVID patients.

**Where to access**  
* CSV column names: ``actuals.icuBeds``
* JSON file fields: ``actuals.icuBeds``, ``actualsTimeseries.*.icuBeds``

### Hospital Beds

  Information about acute bed utilization details.

Fields:
 * capacity - Current staffed acute bed capacity.
 * currentUsageTotal - Total number of acute beds currently in use
 * currentUsageCovid - Number of acute beds currently in use by COVID patients.
 * weeklyCovidAdmissions - Number of COVID patients admitted in the past week.

**Where to access**  
* CSV column names: ``actuals.hospitalBeds``
* JSON file fields: ``actuals.hospitalBeds``, ``actualsTimeseries.*.hospitalBeds``

### Icu Capacity Ratio

  Ratio of staffed intensive care unit (ICU) beds that are currently in use.

**Where to access**  
* CSV column names: ``metrics.icuCapacityRatio``
* JSON file fields: ``metrics.icuCapacityRatio``, ``metricsTimeseries.*.icuCapacityRatio``

### Beds With Covid Patients Ratio

  Ratio of staffed hospital beds that are currently in use by COVID patients. For counties, this is calculated using health service area data for the corresponding area. For more information on HSAs see the [Health Service Area section](https://apidocs.covidactnow.org/data-definitions/#health-service-areas)

**Where to access**  
* CSV column names: ``metrics.bedsWithCovidPatientsRatio``
* JSON file fields: ``metrics.bedsWithCovidPatientsRatio``, ``metricsTimeseries.*.bedsWithCovidPatientsRatio``

### Weekly Covid Admissions Per 100k Population

  Number of COVID patients per 100k population admitted in the past week. For counties, this is calculated using health service area data for the corresponding area. For more information on HSAs see the [Health Service Area section](https://apidocs.covidactnow.org/data-definitions/#health-service-areas)

**Where to access**  
* CSV column names: ``metrics.weeklyCovidAdmissionsPer100k``
* JSON file fields: ``metrics.weeklyCovidAdmissionsPer100k``, ``metricsTimeseries.*.weeklyCovidAdmissionsPer100k``

County- and metro-level hospitalization and ICU actuals are calculated from facility-level data which, suppresses near-zero values due to privacy concerns. As a result, this data may undercount real values, particularly at low levels.

## Vaccinations

### Vaccines Distributed

  Number of vaccine doses distributed.

**Where to access**  
* CSV column names: ``actuals.vaccinesDistributed``
* JSON file fields: ``actuals.vaccinesDistributed``, ``actualsTimeseries.*.vaccinesDistributed``

### Vaccinations Initiated

  Number of vaccinations initiated.

This value may vary by type of vaccine, but for Moderna and Pfizer this indicates
number of people vaccinated with the first dose.

**Where to access**  
* CSV column names: ``actuals.vaccinationsInitiated``
* JSON file fields: ``actuals.vaccinationsInitiated``, ``actualsTimeseries.*.vaccinationsInitiated``

### Vaccinations Completed

  Number of vaccinations completed.

This value may vary by type of vaccine, but for Moderna and Pfizer this indicates
number of people vaccinated with both the first and second dose.

**Where to access**  
* CSV column names: ``actuals.vaccinationsCompleted``
* JSON file fields: ``actuals.vaccinationsCompleted``, ``actualsTimeseries.*.vaccinationsCompleted``

### Vaccinations Additional Dose

  Number of individuals who are fully vaccinated and have received a booster (or additional) dose.

**Where to access**  
* CSV column names: ``actuals.vaccinationsAdditionalDose``
* JSON file fields: ``actuals.vaccinationsAdditionalDose``, ``actualsTimeseries.*.vaccinationsAdditionalDose``

### Vaccines Administered

  Total number of vaccine doses administered.

**Where to access**  
* CSV column names: ``actuals.vaccinesAdministered``
* JSON file fields: ``actuals.vaccinesAdministered``, ``actualsTimeseries.*.vaccinesAdministered``

### Vaccines Administered Demographics

  Demographic distributions for administered vaccines.

**Where to access**  
* JSON file fields: ``actuals.vaccinesAdministeredDemographics``

### Vaccinations Initiated Demographics

  Demographic distributions for initiated vaccinations.

**Where to access**  
* JSON file fields: ``actuals.vaccinationsInitiatedDemographics``

### Vaccinations Initiated Ratio

  Ratio of population that has initiated vaccination.

**Where to access**  
* CSV column names: ``metrics.vaccinationsInitiatedRatio``
* JSON file fields: ``metrics.vaccinationsInitiatedRatio``, ``metricsTimeseries.*.vaccinationsInitiatedRatio``

### Vaccinations Completed Ratio

  Ratio of population that has completed vaccination.

**Where to access**  
* CSV column names: ``metrics.vaccinationsCompletedRatio``
* JSON file fields: ``metrics.vaccinationsCompletedRatio``, ``metricsTimeseries.*.vaccinationsCompletedRatio``

### Vaccinations Additional Dose Ratio

  Ratio of population that are fully vaccinated and have received a booster (or additional) dose.

**Where to access**  
* CSV column names: ``metrics.vaccinationsAdditionalDoseRatio``
* JSON file fields: ``metrics.vaccinationsAdditionalDoseRatio``, ``metricsTimeseries.*.vaccinationsAdditionalDoseRatio``



## Deaths

### Deaths

  Cumulative deaths that are suspected or confirmed to have been caused by COVID-19.

**Where to access**  
* CSV column names: ``actuals.deaths``
* JSON file fields: ``actuals.deaths``, ``actualsTimeseries.*.deaths``


## Community Levels

  Community level for region.

  See https://www.cdc.gov/coronavirus/2019-ncov/science/community-levels.html
  for details about how the Community Level is calculated and should be
  interpreted.

  The values correspond to the following levels:

  | API value  | Community Level |
  | ---------- | --------------- |
  | 0 | Low |
  | 1 | Medium |
  | 2 | High |

  Note that we provide two versions of the Community Level. One is called
  `canCommunityLevel` which is calculated using CAN's data sources and is
  available for states, counties, and metros. It is updated daily though
  depends on hospital data which may only update weekly for counties. The
  other is called `cdcCommunityLevel` and is the raw Community Level published
  by the CDC. It is only available for counties and is updated on a weekly
  basis.

**Where to access**  
* CSV column names: ``communityLevels.canCommunityLevel``
* JSON file fields: ``communityLevels.canCommunityLevel``, ``communityLevelsTimeseries.*.canCommunityLevel``

  and 

* CSV column names: ``communityLevels.cdcCommunityLevel``
* JSON file fields: ``communityLevels.cdcCommunityLevel``, ``communityLevelsTimeseries.*.cdcCommunityLevel``


## Health Service Areas

  A Health Service area (HSA) is a collection of one or more contiguous counties which are relatively self-contained with respect to hospital care. HSAs are used when calculating county-level hospital metrics in order to correct for instances where an individual county does not have any, or has few healthcare facilities within its own borders. For more information see https://seer.cancer.gov/seerstat/variables/countyattribs/hsa.html.

  The source for our county to HSA mappings is [`cdc_hsa_mapping.csv`](https://raw.githubusercontent.com/covid-projections/covid-data-model/main/data/misc/cdc_hsa_mapping.csv) which follows the HSA definitions used by the CDC in their [COVID-19 Community Levels](https://www.cdc.gov/coronavirus/2019-ncov/your-health/covid-by-county.html). HSA populations are calculated as the sum of the component county populations.
