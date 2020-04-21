# Covid Act Now V1 API

## Currently available API functionality

The Covid Act Now API provides the same data that powers [covidactnow.org](https://covidactnow.org) but in an easily digestible, machine readable format, intended for consumption by other COVID websites, models, and tools.

### Update frequency

Data is updated every day, typically around midnight US Pacific Time.

### License

The data presented in the Covid Act Now API is licensed under [Creative Commons 4.0 By Attribution](https://creativecommons.org/licenses/by/4.0/). You are welcome to share, copy, and redistribute it, as well as adapt it for your own works, we just ask that you provide attribution to the source (as we have done with [our data sources](https://github.com/covid-projections/covid-data-public#date-sources-for-current--future-use)).

### URL

Files are available for download prefixed by: `https://data.covidactnow.org/latest/...`

In order to get data, you must specify the desired intervention, as well as the state or county you wish to get data for. Information on how to specify an intervention and location is available below.

### Specifying an intervention

Forward projections are available for the following scenarios:

    "NO_MITIGATION", "MODERATE_MITIGATION", "HIGH_MITIGATION"

These are what the website refers to as the "no intervention", "social distancing", and "stay at home" interventions.

Additionally the most appropriate static scenario based on the per-state intervention is returned by specifying:

    "SELECTED_MITIGATION"

To get a dynamic forecast that is based on the actually observed effect of mitigations and other factors in a given state, use:

    "OBSERVED_MITIGATION"

**Note: `OBSERVED_MITIGATION` is only available for states, not counties.**

More information about these interventions, including the definitions, references, and actual values used is [available here](https://data.covidactnow.org/Covid_Act_Now_Model_References_and_Assumptions.pdf).

### State projections

Returns projections for the selected state

    # Current actuals + projections + limits
    # e.g. https://data.covidactnow.org/latest/us/states/CA.OBSERVED_MITIGATION.json
    /us/states/<ST>.<INTERVENTION>.json
    
    # Full timeseries data: actuals + projected limits + data for every four days
    # e.g. https://data.covidactnow.org/latest/us/states/CA.OBSERVED_MITIGATION.timeseries.json 
    /us/states/<ST>.<INTERVENTION>.timeseries.json

### State level aggregation

Returns projections for all states

    # Current actuals + projections + limits
    # e.g. https://data.covidactnow.org/latest/us/states.OBSERVED_MITIGATION.json
    /us/states.<INTERVENTION>.json
    
    # Timeseries data
    # e.g. https://data.covidactnow.org/latest/us/states.OBSERVED_MITIGATION.timeseries.json
    /us/states.<INTERVENTION>.timeseries.json

State aggregates are also available as CSV files:
    
    # Current actuals + projections + limits
    # e.g. https://data.covidactnow.org/latest/us/states.OBSERVED_MITIGATION.csv
    /latest/us/states.<INTERVENTION>.csv
    
    # Timeseries data
    # E.G. https://data.covidactnow.org/latest/us/states.OBSERVED_MITIGATION.timeseries.csv
    /latest/us/states.<INTERVENTION>.timeseries.csv

### County projections

Returns projections for the selected county
    
    # Current actuals + projections + limits
    # e.g. https://data.covidactnow.org/latest/us/counties/06077.SELECTED_MITIGATION.json
    /us/counties/<5-DIGIT-FIPS>.<INTERVENTION>.json 

    # Full timeseries data: actuals + projected limits + data for every four days
    # e.g. https://data.covidactnow.org/latest/us/counties/06077.SELECTED_MITIGATION.timeseries.json
    /latest/us/counties/<5-DIGIT-FIPS>.<INTERVENTION>.timeseries.json 
    
**Note: `OBSERVED_MITIGATION` is not available for counties.**

### County level aggregation

Returns projections for all counties

    # Current actuals + projections + limits
    # e.g. https://data.covidactnow.org/latest/us/counties.SELECTED_MITIGATION.json
    /us/counties.<INTERVENTION>.json
    
    # Timeseries data
    # e.g. https://data.covidactnow.org/latest/us/counties.SELECTED_MITIGATION.timeseries.json
    /us/counties.<INTERVENTION>.timeseries.json

County aggregates are also available as CSV files:
    
    # Current actuals + projections + limits
    # e.g. https://data.covidactnow.org/latest/us/counties.SELECTED_MITIGATION.csv
    /latest/us/counties.<INTERVENTION>.csv
    
    # Timeseries data
    # e.g. https://data.covidactnow.org/latest/us/counties.SELECTED_MITIGATION.timeseries.csv
    /latest/us/counties.<INTERVENTION>.timeseries.csv

**Note: `OBSERVED_MITIGATION` is not available for counties.**

### Data format

This is the data format for both states and counties. 

```js
({
  country,
  stateName,
  countyName, // null for states
  fips, // 2 digit for states, 5 digit for counties
  lat, 
  long,
  lastUpdatedDate, // ISO 8601 date string
  actuals: {
    population,
    intervention, // one of (NO_MITIGATION, MODERATE_MITIGATION, HIGH_MITIGATION)
    cumulativeConfirmedCases,
    cumulativeDeaths,
    hospitalBeds: {
      capacity,
      currentUsage, // Coming soon where available, null currently
    }, 
    ICUBeds: hospitalBedsFormat  // Coming soon where available, null currently
  }, 
  projections: {
    totalHospitalBeds: {
      shortageStartDate, // null if no shortage projected
      peakDate,
      peakShortfall
    },
    ICUBeds: totalHospitalBedsFormat, // Coming soon where available, null currently
  },
  timeseries: [{  // Only included when requesting `*.timeseries.json` or `*.timeseries.csv`.
    date,
    hospitalBedsRequired,
    hospitalBedCapacity,
    ICUBedsInUse,
    ICUBedCapacity, // Coming soon where available, null currently
    cumulativeDeaths,
    cumulativeInfected,
  }],
})
```

## Coming soon

### Hospital bed usage (actuals)

### ICU bed data (capacity, projections, and actuals)

### Additional data formats (Shapefiles)

Will return aggregate information above in different file formats.

    /latest/us/states.<INTERVENTION>.{dbf,shp,shx}
    /latest/us/counties.<INTERVENTION>.{dbf,shp,shx}



