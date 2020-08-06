# PredictionTimeseriesRow Schema

```txt
https://data.covidactnow.org/#/definitions/RegionSummaryWithTimeseries/properties/timeseries/items
```

Base model for API output.


| Abstract            | Extensible | Status         | Identifiable | Custom Properties | Additional Properties | Access Restrictions | Defined In                                                   |
| :------------------ | ---------- | -------------- | ------------ | :---------------- | --------------------- | ------------------- | ------------------------------------------------------------ |
| Can be instantiated | No         | Unknown status | No           | Forbidden         | Allowed               | none                | [schemas.json\*](../out/schemas.json "open original schema") |

## items Type

`object` ([PredictionTimeseriesRow](schemas-definitions-predictiontimeseriesrow.md))

# PredictionTimeseriesRow Properties

| Property                                      | Type      | Required | Nullable       | Defined by                                                                                                                                                                                                            |
| :-------------------------------------------- | --------- | -------- | -------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [date](#date)                                 | `string`  | Required | cannot be null | [Covid Act Now API Schemas](schemas-definitions-predictiontimeseriesrow-properties-date.md "https&#x3A;//data.covidactnow.org/#/definitions/PredictionTimeseriesRow/properties/date")                                 |
| [hospitalBedsRequired](#hospitalBedsRequired) | `integer` | Required | cannot be null | [Covid Act Now API Schemas](schemas-definitions-predictiontimeseriesrow-properties-hospitalbedsrequired.md "https&#x3A;//data.covidactnow.org/#/definitions/PredictionTimeseriesRow/properties/hospitalBedsRequired") |
| [hospitalBedCapacity](#hospitalBedCapacity)   | `integer` | Required | cannot be null | [Covid Act Now API Schemas](schemas-definitions-predictiontimeseriesrow-properties-hospitalbedcapacity.md "https&#x3A;//data.covidactnow.org/#/definitions/PredictionTimeseriesRow/properties/hospitalBedCapacity")   |
| [ICUBedsInUse](#ICUBedsInUse)                 | `integer` | Required | cannot be null | [Covid Act Now API Schemas](schemas-definitions-predictiontimeseriesrow-properties-icubedsinuse.md "https&#x3A;//data.covidactnow.org/#/definitions/PredictionTimeseriesRow/properties/ICUBedsInUse")                 |
| [ICUBedCapacity](#ICUBedCapacity)             | `integer` | Required | cannot be null | [Covid Act Now API Schemas](schemas-definitions-predictiontimeseriesrow-properties-icubedcapacity.md "https&#x3A;//data.covidactnow.org/#/definitions/PredictionTimeseriesRow/properties/ICUBedCapacity")             |
| [ventilatorsInUse](#ventilatorsInUse)         | `integer` | Required | cannot be null | [Covid Act Now API Schemas](schemas-definitions-predictiontimeseriesrow-properties-ventilatorsinuse.md "https&#x3A;//data.covidactnow.org/#/definitions/PredictionTimeseriesRow/properties/ventilatorsInUse")         |
| [ventilatorCapacity](#ventilatorCapacity)     | `integer` | Required | cannot be null | [Covid Act Now API Schemas](schemas-definitions-predictiontimeseriesrow-properties-ventilatorcapacity.md "https&#x3A;//data.covidactnow.org/#/definitions/PredictionTimeseriesRow/properties/ventilatorCapacity")     |
| [RtIndicator](#RtIndicator)                   | `number`  | Required | cannot be null | [Covid Act Now API Schemas](schemas-definitions-predictiontimeseriesrow-properties-rtindicator.md "https&#x3A;//data.covidactnow.org/#/definitions/PredictionTimeseriesRow/properties/RtIndicator")                   |
| [RtIndicatorCI90](#RtIndicatorCI90)           | `number`  | Required | cannot be null | [Covid Act Now API Schemas](schemas-definitions-predictiontimeseriesrow-properties-rtindicatorci90.md "https&#x3A;//data.covidactnow.org/#/definitions/PredictionTimeseriesRow/properties/RtIndicatorCI90")           |
| [cumulativeDeaths](#cumulativeDeaths)         | `integer` | Required | cannot be null | [Covid Act Now API Schemas](schemas-definitions-predictiontimeseriesrow-properties-cumulativedeaths.md "https&#x3A;//data.covidactnow.org/#/definitions/PredictionTimeseriesRow/properties/cumulativeDeaths")         |
| [cumulativeInfected](#cumulativeInfected)     | `integer` | Required | cannot be null | [Covid Act Now API Schemas](schemas-definitions-predictiontimeseriesrow-properties-cumulativeinfected.md "https&#x3A;//data.covidactnow.org/#/definitions/PredictionTimeseriesRow/properties/cumulativeInfected")     |
| [currentInfected](#currentInfected)           | `integer` | Required | cannot be null | [Covid Act Now API Schemas](schemas-definitions-predictiontimeseriesrow-properties-currentinfected.md "https&#x3A;//data.covidactnow.org/#/definitions/PredictionTimeseriesRow/properties/currentInfected")           |
| [currentSusceptible](#currentSusceptible)     | `integer` | Required | cannot be null | [Covid Act Now API Schemas](schemas-definitions-predictiontimeseriesrow-properties-currentsusceptible.md "https&#x3A;//data.covidactnow.org/#/definitions/PredictionTimeseriesRow/properties/currentSusceptible")     |
| [currentExposed](#currentExposed)             | `integer` | Required | cannot be null | [Covid Act Now API Schemas](schemas-definitions-predictiontimeseriesrow-properties-currentexposed.md "https&#x3A;//data.covidactnow.org/#/definitions/PredictionTimeseriesRow/properties/currentExposed")             |

## date




`date`

-   is required
-   Type: `string` ([Date](schemas-definitions-predictiontimeseriesrow-properties-date.md))
-   cannot be null
-   defined in: [Covid Act Now API Schemas](schemas-definitions-predictiontimeseriesrow-properties-date.md "https&#x3A;//data.covidactnow.org/#/definitions/PredictionTimeseriesRow/properties/date")

### date Type

`string` ([Date](schemas-definitions-predictiontimeseriesrow-properties-date.md))

### date Constraints

**date**: the string must be a date string, according to [RFC 3339, section 5.6](https://tools.ietf.org/html/rfc3339 "check the specification")

## hospitalBedsRequired

Number of hospital beds projected to be in-use or that were actually in use (if in the past)


`hospitalBedsRequired`

-   is required
-   Type: `integer` ([Hospitalbedsrequired](schemas-definitions-predictiontimeseriesrow-properties-hospitalbedsrequired.md))
-   cannot be null
-   defined in: [Covid Act Now API Schemas](schemas-definitions-predictiontimeseriesrow-properties-hospitalbedsrequired.md "https&#x3A;//data.covidactnow.org/#/definitions/PredictionTimeseriesRow/properties/hospitalBedsRequired")

### hospitalBedsRequired Type

`integer` ([Hospitalbedsrequired](schemas-definitions-predictiontimeseriesrow-properties-hospitalbedsrequired.md))

## hospitalBedCapacity

Number of hospital beds projected to be in-use or actually in use (if in the past)


`hospitalBedCapacity`

-   is required
-   Type: `integer` ([Hospitalbedcapacity](schemas-definitions-predictiontimeseriesrow-properties-hospitalbedcapacity.md))
-   cannot be null
-   defined in: [Covid Act Now API Schemas](schemas-definitions-predictiontimeseriesrow-properties-hospitalbedcapacity.md "https&#x3A;//data.covidactnow.org/#/definitions/PredictionTimeseriesRow/properties/hospitalBedCapacity")

### hospitalBedCapacity Type

`integer` ([Hospitalbedcapacity](schemas-definitions-predictiontimeseriesrow-properties-hospitalbedcapacity.md))

## ICUBedsInUse

Number of ICU beds projected to be in-use or that were actually in use (if in the past)


`ICUBedsInUse`

-   is required
-   Type: `integer` ([Icubedsinuse](schemas-definitions-predictiontimeseriesrow-properties-icubedsinuse.md))
-   cannot be null
-   defined in: [Covid Act Now API Schemas](schemas-definitions-predictiontimeseriesrow-properties-icubedsinuse.md "https&#x3A;//data.covidactnow.org/#/definitions/PredictionTimeseriesRow/properties/ICUBedsInUse")

### ICUBedsInUse Type

`integer` ([Icubedsinuse](schemas-definitions-predictiontimeseriesrow-properties-icubedsinuse.md))

## ICUBedCapacity

Number of ICU beds projected to be in-use or actually in use (if in the past)


`ICUBedCapacity`

-   is required
-   Type: `integer` ([Icubedcapacity](schemas-definitions-predictiontimeseriesrow-properties-icubedcapacity.md))
-   cannot be null
-   defined in: [Covid Act Now API Schemas](schemas-definitions-predictiontimeseriesrow-properties-icubedcapacity.md "https&#x3A;//data.covidactnow.org/#/definitions/PredictionTimeseriesRow/properties/ICUBedCapacity")

### ICUBedCapacity Type

`integer` ([Icubedcapacity](schemas-definitions-predictiontimeseriesrow-properties-icubedcapacity.md))

## ventilatorsInUse

Number of ventilators projected to be in-use.


`ventilatorsInUse`

-   is required
-   Type: `integer` ([Ventilatorsinuse](schemas-definitions-predictiontimeseriesrow-properties-ventilatorsinuse.md))
-   cannot be null
-   defined in: [Covid Act Now API Schemas](schemas-definitions-predictiontimeseriesrow-properties-ventilatorsinuse.md "https&#x3A;//data.covidactnow.org/#/definitions/PredictionTimeseriesRow/properties/ventilatorsInUse")

### ventilatorsInUse Type

`integer` ([Ventilatorsinuse](schemas-definitions-predictiontimeseriesrow-properties-ventilatorsinuse.md))

## ventilatorCapacity

Total ventilator capacity.


`ventilatorCapacity`

-   is required
-   Type: `integer` ([Ventilatorcapacity](schemas-definitions-predictiontimeseriesrow-properties-ventilatorcapacity.md))
-   cannot be null
-   defined in: [Covid Act Now API Schemas](schemas-definitions-predictiontimeseriesrow-properties-ventilatorcapacity.md "https&#x3A;//data.covidactnow.org/#/definitions/PredictionTimeseriesRow/properties/ventilatorCapacity")

### ventilatorCapacity Type

`integer` ([Ventilatorcapacity](schemas-definitions-predictiontimeseriesrow-properties-ventilatorcapacity.md))

## RtIndicator

Historical or Inferred Rt


`RtIndicator`

-   is required
-   Type: `number` ([Rtindicator](schemas-definitions-predictiontimeseriesrow-properties-rtindicator.md))
-   cannot be null
-   defined in: [Covid Act Now API Schemas](schemas-definitions-predictiontimeseriesrow-properties-rtindicator.md "https&#x3A;//data.covidactnow.org/#/definitions/PredictionTimeseriesRow/properties/RtIndicator")

### RtIndicator Type

`number` ([Rtindicator](schemas-definitions-predictiontimeseriesrow-properties-rtindicator.md))

## RtIndicatorCI90

Rt standard deviation


`RtIndicatorCI90`

-   is required
-   Type: `number` ([Rtindicatorci90](schemas-definitions-predictiontimeseriesrow-properties-rtindicatorci90.md))
-   cannot be null
-   defined in: [Covid Act Now API Schemas](schemas-definitions-predictiontimeseriesrow-properties-rtindicatorci90.md "https&#x3A;//data.covidactnow.org/#/definitions/PredictionTimeseriesRow/properties/RtIndicatorCI90")

### RtIndicatorCI90 Type

`number` ([Rtindicatorci90](schemas-definitions-predictiontimeseriesrow-properties-rtindicatorci90.md))

## cumulativeDeaths

Number of cumulative deaths


`cumulativeDeaths`

-   is required
-   Type: `integer` ([Cumulativedeaths](schemas-definitions-predictiontimeseriesrow-properties-cumulativedeaths.md))
-   cannot be null
-   defined in: [Covid Act Now API Schemas](schemas-definitions-predictiontimeseriesrow-properties-cumulativedeaths.md "https&#x3A;//data.covidactnow.org/#/definitions/PredictionTimeseriesRow/properties/cumulativeDeaths")

### cumulativeDeaths Type

`integer` ([Cumulativedeaths](schemas-definitions-predictiontimeseriesrow-properties-cumulativedeaths.md))

## cumulativeInfected

Number of cumulative infections


`cumulativeInfected`

-   is required
-   Type: `integer` ([Cumulativeinfected](schemas-definitions-predictiontimeseriesrow-properties-cumulativeinfected.md))
-   cannot be null
-   defined in: [Covid Act Now API Schemas](schemas-definitions-predictiontimeseriesrow-properties-cumulativeinfected.md "https&#x3A;//data.covidactnow.org/#/definitions/PredictionTimeseriesRow/properties/cumulativeInfected")

### cumulativeInfected Type

`integer` ([Cumulativeinfected](schemas-definitions-predictiontimeseriesrow-properties-cumulativeinfected.md))

## currentInfected

Number of current infections


`currentInfected`

-   is required
-   Type: `integer` ([Currentinfected](schemas-definitions-predictiontimeseriesrow-properties-currentinfected.md))
-   cannot be null
-   defined in: [Covid Act Now API Schemas](schemas-definitions-predictiontimeseriesrow-properties-currentinfected.md "https&#x3A;//data.covidactnow.org/#/definitions/PredictionTimeseriesRow/properties/currentInfected")

### currentInfected Type

`integer` ([Currentinfected](schemas-definitions-predictiontimeseriesrow-properties-currentinfected.md))

## currentSusceptible

Number of people currently susceptible 


`currentSusceptible`

-   is required
-   Type: `integer` ([Currentsusceptible](schemas-definitions-predictiontimeseriesrow-properties-currentsusceptible.md))
-   cannot be null
-   defined in: [Covid Act Now API Schemas](schemas-definitions-predictiontimeseriesrow-properties-currentsusceptible.md "https&#x3A;//data.covidactnow.org/#/definitions/PredictionTimeseriesRow/properties/currentSusceptible")

### currentSusceptible Type

`integer` ([Currentsusceptible](schemas-definitions-predictiontimeseriesrow-properties-currentsusceptible.md))

## currentExposed

Number of people currently exposed


`currentExposed`

-   is required
-   Type: `integer` ([Currentexposed](schemas-definitions-predictiontimeseriesrow-properties-currentexposed.md))
-   cannot be null
-   defined in: [Covid Act Now API Schemas](schemas-definitions-predictiontimeseriesrow-properties-currentexposed.md "https&#x3A;//data.covidactnow.org/#/definitions/PredictionTimeseriesRow/properties/currentExposed")

### currentExposed Type

`integer` ([Currentexposed](schemas-definitions-predictiontimeseriesrow-properties-currentexposed.md))
