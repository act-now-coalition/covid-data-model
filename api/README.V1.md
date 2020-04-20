# Covid Act Now V1 API

## Currently available API functionality

The Covid Act Now API provides the same data that powers [covidactnow.org](https://covidactnow.org) but in an easily digestible, machine readable format, intended for consumption by other COVID websites, models, and tools.

### Update frequency

Data is updated every day, typically around midnight US Pacific Time.

### URL

Files are available for download prefixed by: `https://data.covidactnow.org/latest/...`

In order to get data, you must specify the desired intervention, as well as the state or county you wish to get data for. Information on how to specify an intervention and location is available below.

### Specifying an intervention

Forward projections are available for the following scenarios:

    "NO_MITIGATION", "MODERATE_MITIGATION", "HIGH_MITIGATION"

These roughly correlate to the "no intervention", "social distancing", and "stay at home" interventions.

Additionally the most appropriate static scenario based on the per-state intervention is returned by specifying:

    "SELECTED_MITIGATION"

To get a dynamic forecast that is based on the actually observed effect of mitigations and other factors in a given state, use:

    "OBSERVED_MITIGATION"

**Note: `OBSERVED_MITIGATION` is only available for states, not counties.**

More information about these interventions, including the definitions, references, and actual values used is [available here](https://data.covidactnow.org/Covid_Act_Now_Model_References_and_Assumptions.pdf).

### State projections

    # Current "actual" information + projected limits
    /us/states/<ST>.<INTERVENTION>.json
    # e.g. https://data.covidactnow.org/latest/us/states/CA.OBSERVED_MITIGATION.json
    
    # Full timeseries data: actuals + projected limits + data for every four days
    /us/states/<ST>.<INTERVENTION>.timeseries.json
    # e.g. https://data.covidactnow.org/latest/us/states/CA.OBSERVED_MITIGATION.timeseries.json 

### County projections
    
    # Current "actual" information + projected limits
    /us/counties/<5-DIGIT-FIPS>.<INTERVENTION>.json 
    # e.g. https://data.covidactnow.org/latest/us/counties/06077.SELECTED_MITIGATION.json
    
    # Full timeseries data: actuals + projected limits + data for every four days
    /latest/us/counties/<5-DIGIT-FIPS>.<INTERVENTION>.timeseries.json 
    # e.g. https://data.covidactnow.org/latest/us/counties/06077.SELECTED_MITIGATION.timeseries.json
    
**Note: `OBSERVED_MITIGATION` is not available for counties.**

### Data format:

This is the data format for both states and counties. `timeseries` is only included when requesting `*.timeseries.json`.

    {
      country,
      stateName,
      countyName, // null for states
      fips, // 2 digit for states, 5 digit for counties
      lat, 
      long,
      lastUpdatedDate, // ISO 8601 date string
      actuals: {
        population,
        intervention, // one of (NO_MITIGATION, MODERATE_MITIGATION, stay_at_home)
        cumulativeConfirmedCases,
        cumulativeDeaths,
        hospitalBeds: {
          capacity,
          currentUsage, // Coming soon where available, null currently
        }, 
        ICUBeds: { same as above }  // Coming soon where available, null currently
      }, 
      projections: {
        totalHospitalBeds: {
          shortageStartDate, // null if no shortage projected
          peakDate,
          peakShortfall
        },
        ICUBeds: { same as above }, // Coming soon where available, null currently
      },
      timeseries: [{
        date,
        hospitalBedsRequired,
        hospitalBedCapacity,
        ICUBedsInUse,
        ICUBedCapacity, // Coming soon where availabe, null currently
        cumulativeDeaths,
        cumulativeInfected,
      }],
    };

## Coming soon

Additional V1 API endpoints containing batch versions of the data

### State level aggregation

Will return information about all states.

    # everything except timeseries
    /us/states.<INTERVENTION>.json
    [{stateName:'CA', …}, {stateName:'TX',…}, …]
    
    # just the timeseries
    /us/states.<INTERVENTION>.timeseries.json 
    [{stateName:'CA', timeseries:[…],… }, {stateName:'TX', timeseries:[…], …}, …]

### County level aggregation per state

Will return all the county level data for a given state.

    # everything except timeseries
    /us/counties.json
    [{stateName:'CA', countyName, fips, …}, …]
    
    # just the timeseries
    /us/counties(.intervention).timeseries.json
    [{stateName:'CA', countyName, fips, timeseries:[…],… }, …]

### Additional data formats (CSV, Shapefiles)

Will return aggregate information above in different file formats.

    /latest/us/states.<INTERVENTION>.csv
    /latest/us/states.<INTERVENTION>.timeseries.csv
    /latest/us/states.<INTERVENTION>.{dbf,shp,shx}
    /latest/us/counties.<INTERVENTION>.csv
    /latest/us/counties.<INTERVENTION>.timeseries.csv
    /latest/us/counties.<INTERVENTION>.{dbf,shp,shx}



