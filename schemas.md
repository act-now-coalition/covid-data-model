
###ResourceUsageProjection
Base model for API output.
    
|-------------------|---------|------------------------------------------------------|
| peakShortfall     | integer | Shortfall of resource needed at the peak utilization |
| peakDate          | string  | Date of peak resource utilization                    |
| shortageStartDate | string  | Date when resource shortage begins                   |
    
###Projections
Base model for API output.
    
|-------------------|--------|--------------------------------------------------------|
| totalHospitalBeds |        | Projection about total hospital bed utilization        |
| ICUBeds           |        | Projection about ICU hospital bed utilization          |
| Rt                | number | Inferred Rt                                            |
| RtCI90            | number | Rt 90th percentile confidence interval upper endpoint. |
    
###ResourceUtilization
Base model for API output.
    
|-------------------|---------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| capacity          | integer | *deprecated*: Capacity for resource. In the case of ICUs, this refers to total capacity. For hospitalization this refers to free capacity for COVID patients. This value is calculated by (1 - typicalUsageRate) * totalCapacity * 2.07 |
| totalCapacity     | integer | Total capacity for resource.                                                                                                                                                                                                            |
| currentUsageCovid | integer | Currently used capacity for resource by COVID                                                                                                                                                                                           |
| currentUsageTotal | integer | Currently used capacity for resource by all patients (COVID + Non-COVID)                                                                                                                                                                |
| typicalUsageRate  | number  | Typical used capacity rate for resource. This excludes any COVID usage.                                                                                                                                                                 |
    
###Actuals
Base model for API output.
    
|--------------------------|---------------------------------------------|---------------------------------------------------------------------------------|
| population               | integer                                     | Total population in geographic region [*deprecated*: refer to summary for this] |
| intervention             | string                                      | Name of high-level intervention in-place                                        |
| cumulativeConfirmedCases | integer                                     | Number of confirmed cases so far                                                |
| cumulativePositiveTests  | integer                                     | Number of positive test results to date                                         |
| cumulativeNegativeTests  | integer                                     | Number of negative test results to date                                         |
| cumulativeDeaths         | integer                                     | Number of deaths so far                                                         |
| hospitalBeds             | [ResourceUtilization](#ResourceUtilization) | Base model for API output.                                                      |
| ICUBeds                  | [ResourceUtilization](#ResourceUtilization) | Base model for API output.                                                      |
| contactTracers           | integer                                     | # of Contact Tracers                                                            |
    
###ActualsTimeseriesRow
Base model for API output.
    
|--------------------------|---------------------------------------------|---------------------------------------------------------------------------------|
| population               | integer                                     | Total population in geographic region [*deprecated*: refer to summary for this] |
| intervention             | string                                      | Name of high-level intervention in-place                                        |
| cumulativeConfirmedCases | integer                                     | Number of confirmed cases so far                                                |
| cumulativePositiveTests  | integer                                     | Number of positive test results to date                                         |
| cumulativeNegativeTests  | integer                                     | Number of negative test results to date                                         |
| cumulativeDeaths         | integer                                     | Number of deaths so far                                                         |
| hospitalBeds             | [ResourceUtilization](#ResourceUtilization) | Base model for API output.                                                      |
| ICUBeds                  | [ResourceUtilization](#ResourceUtilization) | Base model for API output.                                                      |
| contactTracers           | integer                                     | # of Contact Tracers                                                            |
| date                     | string                                      | Date of timeseries data point                                                   |
    
###RegionSummary
Base model for API output.
    
|-----------------|-----------------------------|--------------------------------------------------------------------------------------|
| countryName     | string                      |                                                                                      |
| fips            | string                      | Fips Code.  For state level data, 2 characters, for county level data, 5 characters. |
| lat             | number                      | Latitude of point within the state or county                                         |
| long            | number                      | Longitude of point within the state or county                                        |
| stateName       | string                      | The state name                                                                       |
| countyName      | string                      | The county name                                                                      |
| lastUpdatedDate | string                      | Date of latest data                                                                  |
| projections     | [Projections](#Projections) | Base model for API output.                                                           |
| actuals         | [Actuals](#Actuals)         | Base model for API output.                                                           |
| population      | integer                     | Total Population in geographic region.                                               |
    
###PredictionTimeseriesRow
Base model for API output.
    
|----------------------|---------|----------------------------------------------------------------------------------------------|
| date                 | string  | Date of timeseries data point                                                                |
| hospitalBedsRequired | integer | Number of hospital beds projected to be in-use or that were actually in use (if in the past) |
| hospitalBedCapacity  | integer | Number of hospital beds projected to be in-use or actually in use (if in the past)           |
| ICUBedsInUse         | integer | Number of ICU beds projected to be in-use or that were actually in use (if in the past)      |
| ICUBedCapacity       | integer | Number of ICU beds projected to be in-use or actually in use (if in the past)                |
| ventilatorsInUse     | integer | Number of ventilators projected to be in-use.                                                |
| ventilatorCapacity   | integer | Total ventilator capacity.                                                                   |
| RtIndicator          | number  | Historical or Inferred Rt                                                                    |
| RtIndicatorCI90      | number  | Rt standard deviation                                                                        |
| cumulativeDeaths     | integer | Number of cumulative deaths                                                                  |
| cumulativeInfected   | integer | Number of cumulative infections                                                              |
| currentInfected      | integer | Number of current infections                                                                 |
| currentSusceptible   | integer | Number of people currently susceptible                                                       |
| currentExposed       | integer | Number of people currently exposed                                                           |
    
###RegionSummaryWithTimeseries
Base model for API output.
    
|-------------------|-----------------------------|--------------------------------------------------------------------------------------|
| countryName       | string                      |                                                                                      |
| fips              | string                      | Fips Code.  For state level data, 2 characters, for county level data, 5 characters. |
| lat               | number                      | Latitude of point within the state or county                                         |
| long              | number                      | Longitude of point within the state or county                                        |
| stateName         | string                      | The state name                                                                       |
| countyName        | string                      | The county name                                                                      |
| lastUpdatedDate   | string                      | Date of latest data                                                                  |
| projections       | [Projections](#Projections) | Base model for API output.                                                           |
| actuals           | [Actuals](#Actuals)         | Base model for API output.                                                           |
| population        | integer                     | Total Population in geographic region.                                               |
| timeseries        | array                       |                                                                                      |
| actualsTimeseries | array                       |                                                                                      |
    
###PredictionTimeseriesRowWithHeader
Base model for API output.
    
|----------------------|---------|----------------------------------------------------------------------------------------------|
| date                 | string  | Date of timeseries data point                                                                |
| hospitalBedsRequired | integer | Number of hospital beds projected to be in-use or that were actually in use (if in the past) |
| hospitalBedCapacity  | integer | Number of hospital beds projected to be in-use or actually in use (if in the past)           |
| ICUBedsInUse         | integer | Number of ICU beds projected to be in-use or that were actually in use (if in the past)      |
| ICUBedCapacity       | integer | Number of ICU beds projected to be in-use or actually in use (if in the past)                |
| ventilatorsInUse     | integer | Number of ventilators projected to be in-use.                                                |
| ventilatorCapacity   | integer | Total ventilator capacity.                                                                   |
| RtIndicator          | number  | Historical or Inferred Rt                                                                    |
| RtIndicatorCI90      | number  | Rt standard deviation                                                                        |
| cumulativeDeaths     | integer | Number of cumulative deaths                                                                  |
| cumulativeInfected   | integer | Number of cumulative infections                                                              |
| currentInfected      | integer | Number of current infections                                                                 |
| currentSusceptible   | integer | Number of people currently susceptible                                                       |
| currentExposed       | integer | Number of people currently exposed                                                           |
| countryName          | string  |                                                                                              |
| stateName            | string  | The state name                                                                               |
| countyName           | string  | The county name                                                                              |
| intervention         | string  | Name of high-level intervention in-place                                                     |
| fips                 | string  | Fips for State + County. Five character code                                                 |
| lat                  | number  | Latitude of point within the state or county                                                 |
| long                 | number  | Longitude of point within the state or county                                                |
| lastUpdatedDate      | string  | Date of latest data                                                                          |
    