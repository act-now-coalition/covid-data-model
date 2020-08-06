# RegionSummaryWithTimeseries Schema

```txt
https://data.covidactnow.org/#/definitions/AggregateRegionSummaryWithTimeseries/items
```

Base model for API output.


| Abstract            | Extensible | Status         | Identifiable | Custom Properties | Additional Properties | Access Restrictions | Defined In                                                   |
| :------------------ | ---------- | -------------- | ------------ | :---------------- | --------------------- | ------------------- | ------------------------------------------------------------ |
| Can be instantiated | No         | Unknown status | No           | Forbidden         | Allowed               | none                | [schemas.json\*](../out/schemas.json "open original schema") |

## items Type

`object` ([RegionSummaryWithTimeseries](schemas-definitions-regionsummarywithtimeseries.md))

# RegionSummaryWithTimeseries Properties

| Property                                | Type      | Required | Nullable       | Defined by                                                                                                                                                                                                              |
| :-------------------------------------- | --------- | -------- | -------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [countryName](#countryName)             | `string`  | Optional | cannot be null | [Covid Act Now API Schemas](schemas-definitions-regionsummarywithtimeseries-properties-countryname.md "https&#x3A;//data.covidactnow.org/#/definitions/RegionSummaryWithTimeseries/properties/countryName")             |
| [fips](#fips)                           | `string`  | Required | cannot be null | [Covid Act Now API Schemas](schemas-definitions-regionsummarywithtimeseries-properties-fips.md "https&#x3A;//data.covidactnow.org/#/definitions/RegionSummaryWithTimeseries/properties/fips")                           |
| [lat](#lat)                             | `number`  | Required | cannot be null | [Covid Act Now API Schemas](schemas-definitions-regionsummarywithtimeseries-properties-lat.md "https&#x3A;//data.covidactnow.org/#/definitions/RegionSummaryWithTimeseries/properties/lat")                             |
| [long](#long)                           | `number`  | Required | cannot be null | [Covid Act Now API Schemas](schemas-definitions-regionsummarywithtimeseries-properties-long.md "https&#x3A;//data.covidactnow.org/#/definitions/RegionSummaryWithTimeseries/properties/long")                           |
| [stateName](#stateName)                 | `string`  | Required | cannot be null | [Covid Act Now API Schemas](schemas-definitions-regionsummarywithtimeseries-properties-statename.md "https&#x3A;//data.covidactnow.org/#/definitions/RegionSummaryWithTimeseries/properties/stateName")                 |
| [countyName](#countyName)               | `string`  | Optional | cannot be null | [Covid Act Now API Schemas](schemas-definitions-regionsummarywithtimeseries-properties-countyname.md "https&#x3A;//data.covidactnow.org/#/definitions/RegionSummaryWithTimeseries/properties/countyName")               |
| [lastUpdatedDate](#lastUpdatedDate)     | `string`  | Required | cannot be null | [Covid Act Now API Schemas](schemas-definitions-regionsummarywithtimeseries-properties-lastupdateddate.md "https&#x3A;//data.covidactnow.org/#/definitions/RegionSummaryWithTimeseries/properties/lastUpdatedDate")     |
| [projections](#projections)             | `object`  | Required | cannot be null | [Covid Act Now API Schemas](schemas-definitions-projections.md "https&#x3A;//data.covidactnow.org/#/definitions/RegionSummaryWithTimeseries/properties/projections")                                                    |
| [actuals](#actuals)                     | `object`  | Required | cannot be null | [Covid Act Now API Schemas](schemas-definitions-actuals.md "https&#x3A;//data.covidactnow.org/#/definitions/RegionSummaryWithTimeseries/properties/actuals")                                                            |
| [population](#population)               | `integer` | Required | cannot be null | [Covid Act Now API Schemas](schemas-definitions-regionsummarywithtimeseries-properties-population.md "https&#x3A;//data.covidactnow.org/#/definitions/RegionSummaryWithTimeseries/properties/population")               |
| [timeseries](#timeseries)               | `array`   | Required | cannot be null | [Covid Act Now API Schemas](schemas-definitions-regionsummarywithtimeseries-properties-timeseries.md "https&#x3A;//data.covidactnow.org/#/definitions/RegionSummaryWithTimeseries/properties/timeseries")               |
| [actualsTimeseries](#actualsTimeseries) | `array`   | Required | cannot be null | [Covid Act Now API Schemas](schemas-definitions-regionsummarywithtimeseries-properties-actualstimeseries.md "https&#x3A;//data.covidactnow.org/#/definitions/RegionSummaryWithTimeseries/properties/actualsTimeseries") |

## countryName




`countryName`

-   is optional
-   Type: `string` ([Countryname](schemas-definitions-regionsummarywithtimeseries-properties-countryname.md))
-   cannot be null
-   defined in: [Covid Act Now API Schemas](schemas-definitions-regionsummarywithtimeseries-properties-countryname.md "https&#x3A;//data.covidactnow.org/#/definitions/RegionSummaryWithTimeseries/properties/countryName")

### countryName Type

`string` ([Countryname](schemas-definitions-regionsummarywithtimeseries-properties-countryname.md))

### countryName Default Value

The default value is:

```json
"US"
```

## fips

Fips Code.  For state level data, 2 characters, for county level data, 5 characters.


`fips`

-   is required
-   Type: `string` ([Fips](schemas-definitions-regionsummarywithtimeseries-properties-fips.md))
-   cannot be null
-   defined in: [Covid Act Now API Schemas](schemas-definitions-regionsummarywithtimeseries-properties-fips.md "https&#x3A;//data.covidactnow.org/#/definitions/RegionSummaryWithTimeseries/properties/fips")

### fips Type

`string` ([Fips](schemas-definitions-regionsummarywithtimeseries-properties-fips.md))

## lat

Latitude of point within the state or county


`lat`

-   is required
-   Type: `number` ([Lat](schemas-definitions-regionsummarywithtimeseries-properties-lat.md))
-   cannot be null
-   defined in: [Covid Act Now API Schemas](schemas-definitions-regionsummarywithtimeseries-properties-lat.md "https&#x3A;//data.covidactnow.org/#/definitions/RegionSummaryWithTimeseries/properties/lat")

### lat Type

`number` ([Lat](schemas-definitions-regionsummarywithtimeseries-properties-lat.md))

## long

Longitude of point within the state or county


`long`

-   is required
-   Type: `number` ([Long](schemas-definitions-regionsummarywithtimeseries-properties-long.md))
-   cannot be null
-   defined in: [Covid Act Now API Schemas](schemas-definitions-regionsummarywithtimeseries-properties-long.md "https&#x3A;//data.covidactnow.org/#/definitions/RegionSummaryWithTimeseries/properties/long")

### long Type

`number` ([Long](schemas-definitions-regionsummarywithtimeseries-properties-long.md))

## stateName

The state name


`stateName`

-   is required
-   Type: `string` ([Statename](schemas-definitions-regionsummarywithtimeseries-properties-statename.md))
-   cannot be null
-   defined in: [Covid Act Now API Schemas](schemas-definitions-regionsummarywithtimeseries-properties-statename.md "https&#x3A;//data.covidactnow.org/#/definitions/RegionSummaryWithTimeseries/properties/stateName")

### stateName Type

`string` ([Statename](schemas-definitions-regionsummarywithtimeseries-properties-statename.md))

## countyName

The county name


`countyName`

-   is optional
-   Type: `string` ([Countyname](schemas-definitions-regionsummarywithtimeseries-properties-countyname.md))
-   cannot be null
-   defined in: [Covid Act Now API Schemas](schemas-definitions-regionsummarywithtimeseries-properties-countyname.md "https&#x3A;//data.covidactnow.org/#/definitions/RegionSummaryWithTimeseries/properties/countyName")

### countyName Type

`string` ([Countyname](schemas-definitions-regionsummarywithtimeseries-properties-countyname.md))

## lastUpdatedDate

Date of latest data


`lastUpdatedDate`

-   is required
-   Type: `string` ([Lastupdateddate](schemas-definitions-regionsummarywithtimeseries-properties-lastupdateddate.md))
-   cannot be null
-   defined in: [Covid Act Now API Schemas](schemas-definitions-regionsummarywithtimeseries-properties-lastupdateddate.md "https&#x3A;//data.covidactnow.org/#/definitions/RegionSummaryWithTimeseries/properties/lastUpdatedDate")

### lastUpdatedDate Type

`string` ([Lastupdateddate](schemas-definitions-regionsummarywithtimeseries-properties-lastupdateddate.md))

### lastUpdatedDate Constraints

**date**: the string must be a date string, according to [RFC 3339, section 5.6](https://tools.ietf.org/html/rfc3339 "check the specification")

## projections

Base model for API output.


`projections`

-   is required
-   Type: `object` ([Projections](schemas-definitions-projections.md))
-   cannot be null
-   defined in: [Covid Act Now API Schemas](schemas-definitions-projections.md "https&#x3A;//data.covidactnow.org/#/definitions/RegionSummaryWithTimeseries/properties/projections")

### projections Type

`object` ([Projections](schemas-definitions-projections.md))

## actuals

Base model for API output.


`actuals`

-   is required
-   Type: `object` ([Actuals](schemas-definitions-actuals.md))
-   cannot be null
-   defined in: [Covid Act Now API Schemas](schemas-definitions-actuals.md "https&#x3A;//data.covidactnow.org/#/definitions/RegionSummaryWithTimeseries/properties/actuals")

### actuals Type

`object` ([Actuals](schemas-definitions-actuals.md))

## population

Total Population in geographic region.


`population`

-   is required
-   Type: `integer` ([Population](schemas-definitions-regionsummarywithtimeseries-properties-population.md))
-   cannot be null
-   defined in: [Covid Act Now API Schemas](schemas-definitions-regionsummarywithtimeseries-properties-population.md "https&#x3A;//data.covidactnow.org/#/definitions/RegionSummaryWithTimeseries/properties/population")

### population Type

`integer` ([Population](schemas-definitions-regionsummarywithtimeseries-properties-population.md))

### population Constraints

**minimum (exclusive)**: the value of this number must be greater than: `0`

## timeseries




`timeseries`

-   is required
-   Type: `object[]` ([PredictionTimeseriesRow](schemas-definitions-predictiontimeseriesrow.md))
-   cannot be null
-   defined in: [Covid Act Now API Schemas](schemas-definitions-regionsummarywithtimeseries-properties-timeseries.md "https&#x3A;//data.covidactnow.org/#/definitions/RegionSummaryWithTimeseries/properties/timeseries")

### timeseries Type

`object[]` ([PredictionTimeseriesRow](schemas-definitions-predictiontimeseriesrow.md))

## actualsTimeseries




`actualsTimeseries`

-   is required
-   Type: `object[]` ([ActualsTimeseriesRow](schemas-definitions-actualstimeseriesrow.md))
-   cannot be null
-   defined in: [Covid Act Now API Schemas](schemas-definitions-regionsummarywithtimeseries-properties-actualstimeseries.md "https&#x3A;//data.covidactnow.org/#/definitions/RegionSummaryWithTimeseries/properties/actualsTimeseries")

### actualsTimeseries Type

`object[]` ([ActualsTimeseriesRow](schemas-definitions-actualstimeseriesrow.md))
