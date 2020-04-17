# CAN V1 API Draft

# Available Now

## V1 API

https://data.covidactnow.org/latest/

**Specifying an intervention**
Forward projections are available for the following scenarios:

    "NO_MITIGATION", "MODERATE_MITIGATION", "HIGH_MITIGATION"

Additionally the most appropriate static scenario based on the per-state intervention is returned by specifying:

    "SELECTED_MITIGATION"

To get a dynamic forecast that is based on the actually observed effect of mitigations and other factors in a given state, use:

    "OBSERVED_MITIGATION"

“Observer Mitigation” is only available for states, not counties.

**State projections available at:**

    /latest/us/states/ca(.intervention).json
    /latest/us/states/ca(.intervention).timeseries.json
    eg. /latest/us/states/CA.OBSERVED_MITIGATION.json and /latest/us/states/CA.OBSERVED_MITIGATION.timeseries.json 

County projections available at:

    /latest/us/counties/5-digit-fips(.intervention).json 
    /latest/us/counties/5-digit-fips(.intervention).timeseries.json 
    eg. /latest/us/counties/06077.SELECTED_MITIGATION.json and /latest/us/counties/06077.SELECTED_MITIGATION.timeseries.json 

The format is:

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
        intervention, // one of (limited_action, social_distancing, stay_at_home)
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
        ICUBedCapacity, // Coming soon where availabe, null for now
        cumulativeDeaths,
        cumulativeInfected,
      }],
    };



# Coming soon

Additional V1 API endpoints containing batch versions of the data

Across all states:

    /latest/us/states(.intervention).json <- everything except timeseries
    [{stateName:'CA', …}, {stateName:'TX',…}, …]
    /latest/us/states(.intervention).timeseries.json <-  just the timeseries
    [{stateName:'CA', timeseries:[…],… }, {stateName:'TX', timeseries:[…], …}, …]

Across all counties in all states:

    /latest/us/counties.json <- everything except timeseries
    [{stateName:'CA', countyName, fips, …}, …]
    /latest/us/counties(.intervention).timeseries.json <- just the timeseries
    [{stateName:'CA', countyName, fips, timeseries:[…],… }, …]

Additional endpoints containing additional formats of the batch data:


    /latest/us/states.(.intervention).csv
    /latest/us/states.(.intervention).timeseries.csv
    /latest/us/states.(.intervention).* //shapefiles
    /latest/us/counties.(.intervention).csv
    /latest/us/counties.(.intervention).timeseries.csv
    /latest/us/counties.(.intervention).* //shapefiles



