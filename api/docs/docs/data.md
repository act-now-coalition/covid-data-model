---
id: data
title: Data
---

Read more about the data included in the Covid Act Now API.




## Cases

### Cases

  Cumulative confirmed or suspected cases.

**Where to access**  
* CSV column names: ``actuals.cases``
* JSON file fields: ``actuals.cases``, ``actualsTimeseries.cases``


### New Cases

  New confirmed or suspected cases.

New cases are a processed timeseries of cases - summing new cases may not equal
the cumulative case count.

Processing steps:
 1. If a region does not report cases for a period of time, the first day
    cases start reporting again will not be included. This day likely includes
    multiple days worth of cases and can be misleading to the overall series.
 2. Any days with negative new cases are removed.
 3. An outlier detection filter is applied to the timeseries, removing any data points that
    seem improbable given recent case numbers.  Many times this is due to a backfill of
    previously unreported cases.

**Where to access**  
* CSV column names: ``actuals.newCases``
* JSON file fields: ``actuals.newCases``, ``actualsTimeseries.newCases``


### Case Density

  The number of cases per 100k population calculated using a 7-day rolling average.

**Where to access**  
* CSV column names: ``metrics.caseDensity``
* JSON file fields: ``metrics.caseDensity``, ``metricsTimeseries.caseDensity``


### Infection Rate

  R_t, or the estimated number of infections arising from a typical case.

**Where to access**  
* CSV column names: ``metrics.infectionRate``
* JSON file fields: ``metrics.infectionRate``, ``metricsTimeseries.infectionRate``


### Infection Rate C I 9 0

  90th percentile confidence interval upper endpoint of the infection rate.

**Where to access**  
* CSV column names: ``metrics.infectionRateCI90``
* JSON file fields: ``metrics.infectionRateCI90``, ``metricsTimeseries.infectionRateCI90``



    

## Deaths

### Deaths

  Cumulative deaths that are suspected or confirmed to have been caused by COVID-19

**Where to access**  
* CSV column names: ``actuals.deaths``
* JSON file fields: ``actuals.deaths``, ``actualsTimeseries.deaths``




    

## Hospitalizations

### Icu Beds

  Information about ICU bed utilization details.

Fields:
 * capacity - Current staffed ICU bed capacity.
 * currentUsageTotal - Total number of ICU beds currently in use
 * currentUsageCovid - Number of ICU beds currently in use by COVID patients.
 * typicalUsageRate - Typical ICU utilization rate.

**Where to access**  
* CSV column names: ``actuals.icuBeds``
* JSON file fields: ``actuals.icuBeds``, ``actualsTimeseries.icuBeds``


### Hospital Beds

  Information about acute bed utilization details.

Fields:
 * capacity - Current staffed acute bed capacity.
 * currentUsageTotal - Total number of acute beds currently in use
 * currentUsageCovid - Number of acute beds currently in use by COVID patients.
 * typicalUsageRate - Typical acute bed utilization rate.

**Where to access**  
* CSV column names: ``actuals.hospitalBeds``
* JSON file fields: ``actuals.hospitalBeds``, ``actualsTimeseries.hospitalBeds``


### Icu Capacity Ratio

  Ratio of staffed intensive care unit (ICU) beds that are currently in use.

**Where to access**  
* CSV column names: ``metrics.icuCapacityRatio``
* JSON file fields: ``metrics.icuCapacityRatio``, ``metricsTimeseries.icuCapacityRatio``


### Icu Headroom Ratio

  

**Where to access**  
* CSV column names: ``metrics.icuHeadroomRatio``
* JSON file fields: ``metrics.icuHeadroomRatio``, ``metricsTimeseries.icuHeadroomRatio``



    

## Vaccinations

### Vaccines Distributed

  Number of vaccine doses distributed.

**Where to access**  
* CSV column names: ``actuals.vaccinesDistributed``
* JSON file fields: ``actuals.vaccinesDistributed``, ``actualsTimeseries.vaccinesDistributed``


### Vaccinations Initiated

  Number of vaccinations initiated.

This value may vary by type of vaccine, but for Moderna and Pfizer this indicates
number of people vaccinated with the first dose.

**Where to access**  
* CSV column names: ``actuals.vaccinationsInitiated``
* JSON file fields: ``actuals.vaccinationsInitiated``, ``actualsTimeseries.vaccinationsInitiated``


### Vaccinations Completed

  Number of vaccinations completed.

This value may vary by type of vaccine, but for Moderna and Pfizer this indicates
number of people vaccinated with both the first and second dose.

**Where to access**  
* CSV column names: ``actuals.vaccinationsCompleted``
* JSON file fields: ``actuals.vaccinationsCompleted``, ``actualsTimeseries.vaccinationsCompleted``


### Vaccinations Initiated Ratio

  Ratio of population that has initiated vaccination.

**Where to access**  
* CSV column names: ``metrics.vaccinationsInitiatedRatio``
* JSON file fields: ``metrics.vaccinationsInitiatedRatio``, ``metricsTimeseries.vaccinationsInitiatedRatio``


### Vaccinations Completed Ratio

  Ratio of population that has completed vaccination.

**Where to access**  
* CSV column names: ``metrics.vaccinationsCompletedRatio``
* JSON file fields: ``metrics.vaccinationsCompletedRatio``, ``metricsTimeseries.vaccinationsCompletedRatio``



    
