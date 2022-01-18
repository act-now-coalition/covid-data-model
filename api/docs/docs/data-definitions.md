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

**Where to access**  
* CSV column names: ``actuals.hospitalBeds``
* JSON file fields: ``actuals.hospitalBeds``, ``actualsTimeseries.*.hospitalBeds``

### Icu Capacity Ratio

  Ratio of staffed intensive care unit (ICU) beds that are currently in use.

**Where to access**  
* CSV column names: ``metrics.icuCapacityRatio``
* JSON file fields: ``metrics.icuCapacityRatio``, ``metricsTimeseries.*.icuCapacityRatio``



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



