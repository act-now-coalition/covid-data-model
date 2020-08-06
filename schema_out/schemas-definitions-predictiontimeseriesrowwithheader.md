# PredictionTimeseriesRowWithHeader Schema

```txt
https://data.covidactnow.org/#/definitions/AggregateFlattenedTimeseries/items
```

Base model for API output.


| Abstract            | Extensible | Status         | Identifiable | Custom Properties | Additional Properties | Access Restrictions | Defined In                                                   |
| :------------------ | ---------- | -------------- | ------------ | :---------------- | --------------------- | ------------------- | ------------------------------------------------------------ |
| Can be instantiated | No         | Unknown status | No           | Forbidden         | Allowed               | none                | [schemas.json\*](../out/schemas.json "open original schema") |

## items Type

`object` ([PredictionTimeseriesRowWithHeader](schemas-definitions-predictiontimeseriesrowwithheader.md))

# PredictionTimeseriesRowWithHeader Properties

| Property                                      | Type      | Required | Nullable       | Defined by                                                                                                                                                                                                                                |
| :-------------------------------------------- | --------- | -------- | -------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [date](#date)                                 | `string`  | Required | cannot be null | [Covid Act Now API Schemas](schemas-definitions-predictiontimeseriesrowwithheader-properties-date.md "https&#x3A;//data.covidactnow.org/#/definitions/PredictionTimeseriesRowWithHeader/properties/date")                                 |
| [hospitalBedsRequired](#hospitalBedsRequired) | `integer` | Required | cannot be null | [Covid Act Now API Schemas](schemas-definitions-predictiontimeseriesrowwithheader-properties-hospitalbedsrequired.md "https&#x3A;//data.covidactnow.org/#/definitions/PredictionTimeseriesRowWithHeader/properties/hospitalBedsRequired") |
| [hospitalBedCapacity](#hospitalBedCapacity)   | `integer` | Required | cannot be null | [Covid Act Now API Schemas](schemas-definitions-predictiontimeseriesrowwithheader-properties-hospitalbedcapacity.md "https&#x3A;//data.covidactnow.org/#/definitions/PredictionTimeseriesRowWithHeader/properties/hospitalBedCapacity")   |
| [ICUBedsInUse](#ICUBedsInUse)                 | `integer` | Required | cannot be null | [Covid Act Now API Schemas](schemas-definitions-predictiontimeseriesrowwithheader-properties-icubedsinuse.md "https&#x3A;//data.covidactnow.org/#/definitions/PredictionTimeseriesRowWithHeader/properties/ICUBedsInUse")                 |
| [ICUBedCapacity](#ICUBedCapacity)             | `integer` | Required | cannot be null | [Covid Act Now API Schemas](schemas-definitions-predictiontimeseriesrowwithheader-properties-icubedcapacity.md "https&#x3A;//data.covidactnow.org/#/definitions/PredictionTimeseriesRowWithHeader/properties/ICUBedCapacity")             |
| [ventilatorsInUse](#ventilatorsInUse)         | `integer` | Required | cannot be null | [Covid Act Now API Schemas](schemas-definitions-predictiontimeseriesrowwithheader-properties-ventilatorsinuse.md "https&#x3A;//data.covidactnow.org/#/definitions/PredictionTimeseriesRowWithHeader/properties/ventilatorsInUse")         |
| [ventilatorCapacity](#ventilatorCapacity)     | `integer` | Required | cannot be null | [Covid Act Now API Schemas](schemas-definitions-predictiontimeseriesrowwithheader-properties-ventilatorcapacity.md "https&#x3A;//data.covidactnow.org/#/definitions/PredictionTimeseriesRowWithHeader/properties/ventilatorCapacity")     |
| [RtIndicator](#RtIndicator)                   | `number`  | Required | cannot be null | [Covid Act Now API Schemas](schemas-definitions-predictiontimeseriesrowwithheader-properties-rtindicator.md "https&#x3A;//data.covidactnow.org/#/definitions/PredictionTimeseriesRowWithHeader/properties/RtIndicator")                   |
| [RtIndicatorCI90](#RtIndicatorCI90)           | `number`  | Required | cannot be null | [Covid Act Now API Schemas](schemas-definitions-predictiontimeseriesrowwithheader-properties-rtindicatorci90.md "https&#x3A;//data.covidactnow.org/#/definitions/PredictionTimeseriesRowWithHeader/properties/RtIndicatorCI90")           |
| [cumulativeDeaths](#cumulativeDeaths)         | `integer` | Required | cannot be null | [Covid Act Now API Schemas](schemas-definitions-predictiontimeseriesrowwithheader-properties-cumulativedeaths.md "https&#x3A;//data.covidactnow.org/#/definitions/PredictionTimeseriesRowWithHeader/properties/cumulativeDeaths")         |
| [cumulativeInfected](#cumulativeInfected)     | `integer` | Required | cannot be null | [Covid Act Now API Schemas](schemas-definitions-predictiontimeseriesrowwithheader-properties-cumulativeinfected.md "https&#x3A;//data.covidactnow.org/#/definitions/PredictionTimeseriesRowWithHeader/properties/cumulativeInfected")     |
| [currentInfected](#currentInfected)           | `integer` | Required | cannot be null | [Covid Act Now API Schemas](schemas-definitions-predictiontimeseriesrowwithheader-properties-currentinfected.md "https&#x3A;//data.covidactnow.org/#/definitions/PredictionTimeseriesRowWithHeader/properties/currentInfected")           |
| [currentSusceptible](#currentSusceptible)     | `integer` | Required | cannot be null | [Covid Act Now API Schemas](schemas-definitions-predictiontimeseriesrowwithheader-properties-currentsusceptible.md "https&#x3A;//data.covidactnow.org/#/definitions/PredictionTimeseriesRowWithHeader/properties/currentSusceptible")     |
| [currentExposed](#currentExposed)             | `integer` | Required | cannot be null | [Covid Act Now API Schemas](schemas-definitions-predictiontimeseriesrowwithheader-properties-currentexposed.md "https&#x3A;//data.covidactnow.org/#/definitions/PredictionTimeseriesRowWithHeader/properties/currentExposed")             |
| [countryName](#countryName)                   | `string`  | Optional | cannot be null | [Covid Act Now API Schemas](schemas-definitions-predictiontimeseriesrowwithheader-properties-countryname.md "https&#x3A;//data.covidactnow.org/#/definitions/PredictionTimeseriesRowWithHeader/properties/countryName")                   |
| [stateName](#stateName)                       | `string`  | Required | cannot be null | [Covid Act Now API Schemas](schemas-definitions-predictiontimeseriesrowwithheader-properties-statename.md "https&#x3A;//data.covidactnow.org/#/definitions/PredictionTimeseriesRowWithHeader/properties/stateName")                       |
| [countyName](#countyName)                     | `string`  | Required | cannot be null | [Covid Act Now API Schemas](schemas-definitions-predictiontimeseriesrowwithheader-properties-countyname.md "https&#x3A;//data.covidactnow.org/#/definitions/PredictionTimeseriesRowWithHeader/properties/countyName")                     |
| [intervention](#intervention)                 | `string`  | Required | cannot be null | [Covid Act Now API Schemas](schemas-definitions-predictiontimeseriesrowwithheader-properties-intervention.md "https&#x3A;//data.covidactnow.org/#/definitions/PredictionTimeseriesRowWithHeader/properties/intervention")                 |
| [fips](#fips)                                 | `string`  | Required | cannot be null | [Covid Act Now API Schemas](schemas-definitions-predictiontimeseriesrowwithheader-properties-fips.md "https&#x3A;//data.covidactnow.org/#/definitions/PredictionTimeseriesRowWithHeader/properties/fips")                                 |
| [lat](#lat)                                   | `number`  | Required | cannot be null | [Covid Act Now API Schemas](schemas-definitions-predictiontimeseriesrowwithheader-properties-lat.md "https&#x3A;//data.covidactnow.org/#/definitions/PredictionTimeseriesRowWithHeader/properties/lat")                                   |
| [long](#long)                                 | `number`  | Required | cannot be null | [Covid Act Now API Schemas](schemas-definitions-predictiontimeseriesrowwithheader-properties-long.md "https&#x3A;//data.covidactnow.org/#/definitions/PredictionTimeseriesRowWithHeader/properties/long")                                 |
| [lastUpdatedDate](#lastUpdatedDate)           | `string`  | Required | cannot be null | [Covid Act Now API Schemas](schemas-definitions-predictiontimeseriesrowwithheader-properties-lastupdateddate.md "https&#x3A;//data.covidactnow.org/#/definitions/PredictionTimeseriesRowWithHeader/properties/lastUpdatedDate")           |

## date




`date`

-   is required
-   Type: `string` ([Date](schemas-definitions-predictiontimeseriesrowwithheader-properties-date.md))
-   cannot be null
-   defined in: [Covid Act Now API Schemas](schemas-definitions-predictiontimeseriesrowwithheader-properties-date.md "https&#x3A;//data.covidactnow.org/#/definitions/PredictionTimeseriesRowWithHeader/properties/date")

### date Type

`string` ([Date](schemas-definitions-predictiontimeseriesrowwithheader-properties-date.md))

### date Constraints

**date**: the string must be a date string, according to [RFC 3339, section 5.6](https://tools.ietf.org/html/rfc3339 "check the specification")

## hospitalBedsRequired

Number of hospital beds projected to be in-use or that were actually in use (if in the past)


`hospitalBedsRequired`

-   is required
-   Type: `integer` ([Hospitalbedsrequired](schemas-definitions-predictiontimeseriesrowwithheader-properties-hospitalbedsrequired.md))
-   cannot be null
-   defined in: [Covid Act Now API Schemas](schemas-definitions-predictiontimeseriesrowwithheader-properties-hospitalbedsrequired.md "https&#x3A;//data.covidactnow.org/#/definitions/PredictionTimeseriesRowWithHeader/properties/hospitalBedsRequired")

### hospitalBedsRequired Type

`integer` ([Hospitalbedsrequired](schemas-definitions-predictiontimeseriesrowwithheader-properties-hospitalbedsrequired.md))

## hospitalBedCapacity

Number of hospital beds projected to be in-use or actually in use (if in the past)


`hospitalBedCapacity`

-   is required
-   Type: `integer` ([Hospitalbedcapacity](schemas-definitions-predictiontimeseriesrowwithheader-properties-hospitalbedcapacity.md))
-   cannot be null
-   defined in: [Covid Act Now API Schemas](schemas-definitions-predictiontimeseriesrowwithheader-properties-hospitalbedcapacity.md "https&#x3A;//data.covidactnow.org/#/definitions/PredictionTimeseriesRowWithHeader/properties/hospitalBedCapacity")

### hospitalBedCapacity Type

`integer` ([Hospitalbedcapacity](schemas-definitions-predictiontimeseriesrowwithheader-properties-hospitalbedcapacity.md))

## ICUBedsInUse

Number of ICU beds projected to be in-use or that were actually in use (if in the past)


`ICUBedsInUse`

-   is required
-   Type: `integer` ([Icubedsinuse](schemas-definitions-predictiontimeseriesrowwithheader-properties-icubedsinuse.md))
-   cannot be null
-   defined in: [Covid Act Now API Schemas](schemas-definitions-predictiontimeseriesrowwithheader-properties-icubedsinuse.md "https&#x3A;//data.covidactnow.org/#/definitions/PredictionTimeseriesRowWithHeader/properties/ICUBedsInUse")

### ICUBedsInUse Type

`integer` ([Icubedsinuse](schemas-definitions-predictiontimeseriesrowwithheader-properties-icubedsinuse.md))

## ICUBedCapacity

Number of ICU beds projected to be in-use or actually in use (if in the past)


`ICUBedCapacity`

-   is required
-   Type: `integer` ([Icubedcapacity](schemas-definitions-predictiontimeseriesrowwithheader-properties-icubedcapacity.md))
-   cannot be null
-   defined in: [Covid Act Now API Schemas](schemas-definitions-predictiontimeseriesrowwithheader-properties-icubedcapacity.md "https&#x3A;//data.covidactnow.org/#/definitions/PredictionTimeseriesRowWithHeader/properties/ICUBedCapacity")

### ICUBedCapacity Type

`integer` ([Icubedcapacity](schemas-definitions-predictiontimeseriesrowwithheader-properties-icubedcapacity.md))

## ventilatorsInUse

Number of ventilators projected to be in-use.


`ventilatorsInUse`

-   is required
-   Type: `integer` ([Ventilatorsinuse](schemas-definitions-predictiontimeseriesrowwithheader-properties-ventilatorsinuse.md))
-   cannot be null
-   defined in: [Covid Act Now API Schemas](schemas-definitions-predictiontimeseriesrowwithheader-properties-ventilatorsinuse.md "https&#x3A;//data.covidactnow.org/#/definitions/PredictionTimeseriesRowWithHeader/properties/ventilatorsInUse")

### ventilatorsInUse Type

`integer` ([Ventilatorsinuse](schemas-definitions-predictiontimeseriesrowwithheader-properties-ventilatorsinuse.md))

## ventilatorCapacity

Total ventilator capacity.


`ventilatorCapacity`

-   is required
-   Type: `integer` ([Ventilatorcapacity](schemas-definitions-predictiontimeseriesrowwithheader-properties-ventilatorcapacity.md))
-   cannot be null
-   defined in: [Covid Act Now API Schemas](schemas-definitions-predictiontimeseriesrowwithheader-properties-ventilatorcapacity.md "https&#x3A;//data.covidactnow.org/#/definitions/PredictionTimeseriesRowWithHeader/properties/ventilatorCapacity")

### ventilatorCapacity Type

`integer` ([Ventilatorcapacity](schemas-definitions-predictiontimeseriesrowwithheader-properties-ventilatorcapacity.md))

## RtIndicator

Historical or Inferred Rt


`RtIndicator`

-   is required
-   Type: `number` ([Rtindicator](schemas-definitions-predictiontimeseriesrowwithheader-properties-rtindicator.md))
-   cannot be null
-   defined in: [Covid Act Now API Schemas](schemas-definitions-predictiontimeseriesrowwithheader-properties-rtindicator.md "https&#x3A;//data.covidactnow.org/#/definitions/PredictionTimeseriesRowWithHeader/properties/RtIndicator")

### RtIndicator Type

`number` ([Rtindicator](schemas-definitions-predictiontimeseriesrowwithheader-properties-rtindicator.md))

## RtIndicatorCI90

Rt standard deviation


`RtIndicatorCI90`

-   is required
-   Type: `number` ([Rtindicatorci90](schemas-definitions-predictiontimeseriesrowwithheader-properties-rtindicatorci90.md))
-   cannot be null
-   defined in: [Covid Act Now API Schemas](schemas-definitions-predictiontimeseriesrowwithheader-properties-rtindicatorci90.md "https&#x3A;//data.covidactnow.org/#/definitions/PredictionTimeseriesRowWithHeader/properties/RtIndicatorCI90")

### RtIndicatorCI90 Type

`number` ([Rtindicatorci90](schemas-definitions-predictiontimeseriesrowwithheader-properties-rtindicatorci90.md))

## cumulativeDeaths

Number of cumulative deaths


`cumulativeDeaths`

-   is required
-   Type: `integer` ([Cumulativedeaths](schemas-definitions-predictiontimeseriesrowwithheader-properties-cumulativedeaths.md))
-   cannot be null
-   defined in: [Covid Act Now API Schemas](schemas-definitions-predictiontimeseriesrowwithheader-properties-cumulativedeaths.md "https&#x3A;//data.covidactnow.org/#/definitions/PredictionTimeseriesRowWithHeader/properties/cumulativeDeaths")

### cumulativeDeaths Type

`integer` ([Cumulativedeaths](schemas-definitions-predictiontimeseriesrowwithheader-properties-cumulativedeaths.md))

## cumulativeInfected

Number of cumulative infections


`cumulativeInfected`

-   is required
-   Type: `integer` ([Cumulativeinfected](schemas-definitions-predictiontimeseriesrowwithheader-properties-cumulativeinfected.md))
-   cannot be null
-   defined in: [Covid Act Now API Schemas](schemas-definitions-predictiontimeseriesrowwithheader-properties-cumulativeinfected.md "https&#x3A;//data.covidactnow.org/#/definitions/PredictionTimeseriesRowWithHeader/properties/cumulativeInfected")

### cumulativeInfected Type

`integer` ([Cumulativeinfected](schemas-definitions-predictiontimeseriesrowwithheader-properties-cumulativeinfected.md))

## currentInfected

Number of current infections


`currentInfected`

-   is required
-   Type: `integer` ([Currentinfected](schemas-definitions-predictiontimeseriesrowwithheader-properties-currentinfected.md))
-   cannot be null
-   defined in: [Covid Act Now API Schemas](schemas-definitions-predictiontimeseriesrowwithheader-properties-currentinfected.md "https&#x3A;//data.covidactnow.org/#/definitions/PredictionTimeseriesRowWithHeader/properties/currentInfected")

### currentInfected Type

`integer` ([Currentinfected](schemas-definitions-predictiontimeseriesrowwithheader-properties-currentinfected.md))

## currentSusceptible

Number of people currently susceptible 


`currentSusceptible`

-   is required
-   Type: `integer` ([Currentsusceptible](schemas-definitions-predictiontimeseriesrowwithheader-properties-currentsusceptible.md))
-   cannot be null
-   defined in: [Covid Act Now API Schemas](schemas-definitions-predictiontimeseriesrowwithheader-properties-currentsusceptible.md "https&#x3A;//data.covidactnow.org/#/definitions/PredictionTimeseriesRowWithHeader/properties/currentSusceptible")

### currentSusceptible Type

`integer` ([Currentsusceptible](schemas-definitions-predictiontimeseriesrowwithheader-properties-currentsusceptible.md))

## currentExposed

Number of people currently exposed


`currentExposed`

-   is required
-   Type: `integer` ([Currentexposed](schemas-definitions-predictiontimeseriesrowwithheader-properties-currentexposed.md))
-   cannot be null
-   defined in: [Covid Act Now API Schemas](schemas-definitions-predictiontimeseriesrowwithheader-properties-currentexposed.md "https&#x3A;//data.covidactnow.org/#/definitions/PredictionTimeseriesRowWithHeader/properties/currentExposed")

### currentExposed Type

`integer` ([Currentexposed](schemas-definitions-predictiontimeseriesrowwithheader-properties-currentexposed.md))

## countryName




`countryName`

-   is optional
-   Type: `string` ([Countryname](schemas-definitions-predictiontimeseriesrowwithheader-properties-countryname.md))
-   cannot be null
-   defined in: [Covid Act Now API Schemas](schemas-definitions-predictiontimeseriesrowwithheader-properties-countryname.md "https&#x3A;//data.covidactnow.org/#/definitions/PredictionTimeseriesRowWithHeader/properties/countryName")

### countryName Type

`string` ([Countryname](schemas-definitions-predictiontimeseriesrowwithheader-properties-countryname.md))

### countryName Default Value

The default value is:

```json
"US"
```

## stateName

The state name


`stateName`

-   is required
-   Type: `string` ([Statename](schemas-definitions-predictiontimeseriesrowwithheader-properties-statename.md))
-   cannot be null
-   defined in: [Covid Act Now API Schemas](schemas-definitions-predictiontimeseriesrowwithheader-properties-statename.md "https&#x3A;//data.covidactnow.org/#/definitions/PredictionTimeseriesRowWithHeader/properties/stateName")

### stateName Type

`string` ([Statename](schemas-definitions-predictiontimeseriesrowwithheader-properties-statename.md))

## countyName

The county name


`countyName`

-   is required
-   Type: `string` ([Countyname](schemas-definitions-predictiontimeseriesrowwithheader-properties-countyname.md))
-   cannot be null
-   defined in: [Covid Act Now API Schemas](schemas-definitions-predictiontimeseriesrowwithheader-properties-countyname.md "https&#x3A;//data.covidactnow.org/#/definitions/PredictionTimeseriesRowWithHeader/properties/countyName")

### countyName Type

`string` ([Countyname](schemas-definitions-predictiontimeseriesrowwithheader-properties-countyname.md))

## intervention

Name of high-level intervention in-place


`intervention`

-   is required
-   Type: `string` ([Intervention](schemas-definitions-predictiontimeseriesrowwithheader-properties-intervention.md))
-   cannot be null
-   defined in: [Covid Act Now API Schemas](schemas-definitions-predictiontimeseriesrowwithheader-properties-intervention.md "https&#x3A;//data.covidactnow.org/#/definitions/PredictionTimeseriesRowWithHeader/properties/intervention")

### intervention Type

`string` ([Intervention](schemas-definitions-predictiontimeseriesrowwithheader-properties-intervention.md))

## fips

Fips for State + County. Five character code


`fips`

-   is required
-   Type: `string` ([Fips](schemas-definitions-predictiontimeseriesrowwithheader-properties-fips.md))
-   cannot be null
-   defined in: [Covid Act Now API Schemas](schemas-definitions-predictiontimeseriesrowwithheader-properties-fips.md "https&#x3A;//data.covidactnow.org/#/definitions/PredictionTimeseriesRowWithHeader/properties/fips")

### fips Type

`string` ([Fips](schemas-definitions-predictiontimeseriesrowwithheader-properties-fips.md))

## lat

Latitude of point within the state or county


`lat`

-   is required
-   Type: `number` ([Lat](schemas-definitions-predictiontimeseriesrowwithheader-properties-lat.md))
-   cannot be null
-   defined in: [Covid Act Now API Schemas](schemas-definitions-predictiontimeseriesrowwithheader-properties-lat.md "https&#x3A;//data.covidactnow.org/#/definitions/PredictionTimeseriesRowWithHeader/properties/lat")

### lat Type

`number` ([Lat](schemas-definitions-predictiontimeseriesrowwithheader-properties-lat.md))

## long

Longitude of point within the state or county


`long`

-   is required
-   Type: `number` ([Long](schemas-definitions-predictiontimeseriesrowwithheader-properties-long.md))
-   cannot be null
-   defined in: [Covid Act Now API Schemas](schemas-definitions-predictiontimeseriesrowwithheader-properties-long.md "https&#x3A;//data.covidactnow.org/#/definitions/PredictionTimeseriesRowWithHeader/properties/long")

### long Type

`number` ([Long](schemas-definitions-predictiontimeseriesrowwithheader-properties-long.md))

## lastUpdatedDate

Date of latest data


`lastUpdatedDate`

-   is required
-   Type: `string` ([Lastupdateddate](schemas-definitions-predictiontimeseriesrowwithheader-properties-lastupdateddate.md))
-   cannot be null
-   defined in: [Covid Act Now API Schemas](schemas-definitions-predictiontimeseriesrowwithheader-properties-lastupdateddate.md "https&#x3A;//data.covidactnow.org/#/definitions/PredictionTimeseriesRowWithHeader/properties/lastUpdatedDate")

### lastUpdatedDate Type

`string` ([Lastupdateddate](schemas-definitions-predictiontimeseriesrowwithheader-properties-lastupdateddate.md))

### lastUpdatedDate Constraints

**date**: the string must be a date string, according to [RFC 3339, section 5.6](https://tools.ietf.org/html/rfc3339 "check the specification")
