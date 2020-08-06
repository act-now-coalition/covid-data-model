# ActualsTimeseriesRow Schema

```txt
https://data.covidactnow.org/#/definitions/RegionSummaryWithTimeseries/properties/actualsTimeseries/items
```

Base model for API output.


| Abstract            | Extensible | Status         | Identifiable | Custom Properties | Additional Properties | Access Restrictions | Defined In                                                   |
| :------------------ | ---------- | -------------- | ------------ | :---------------- | --------------------- | ------------------- | ------------------------------------------------------------ |
| Can be instantiated | No         | Unknown status | No           | Forbidden         | Allowed               | none                | [schemas.json\*](../out/schemas.json "open original schema") |

## items Type

`object` ([ActualsTimeseriesRow](schemas-definitions-actualstimeseriesrow.md))

# ActualsTimeseriesRow Properties

| Property                                              | Type      | Required | Nullable       | Defined by                                                                                                                                                                                                              |
| :---------------------------------------------------- | --------- | -------- | -------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [population](#population)                             | `integer` | Required | cannot be null | [Covid Act Now API Schemas](schemas-definitions-actualstimeseriesrow-properties-population.md "https&#x3A;//data.covidactnow.org/#/definitions/ActualsTimeseriesRow/properties/population")                             |
| [intervention](#intervention)                         | `string`  | Required | cannot be null | [Covid Act Now API Schemas](schemas-definitions-actualstimeseriesrow-properties-intervention.md "https&#x3A;//data.covidactnow.org/#/definitions/ActualsTimeseriesRow/properties/intervention")                         |
| [cumulativeConfirmedCases](#cumulativeConfirmedCases) | `integer` | Required | cannot be null | [Covid Act Now API Schemas](schemas-definitions-actualstimeseriesrow-properties-cumulativeconfirmedcases.md "https&#x3A;//data.covidactnow.org/#/definitions/ActualsTimeseriesRow/properties/cumulativeConfirmedCases") |
| [cumulativePositiveTests](#cumulativePositiveTests)   | `integer` | Required | cannot be null | [Covid Act Now API Schemas](schemas-definitions-actualstimeseriesrow-properties-cumulativepositivetests.md "https&#x3A;//data.covidactnow.org/#/definitions/ActualsTimeseriesRow/properties/cumulativePositiveTests")   |
| [cumulativeNegativeTests](#cumulativeNegativeTests)   | `integer` | Required | cannot be null | [Covid Act Now API Schemas](schemas-definitions-actualstimeseriesrow-properties-cumulativenegativetests.md "https&#x3A;//data.covidactnow.org/#/definitions/ActualsTimeseriesRow/properties/cumulativeNegativeTests")   |
| [cumulativeDeaths](#cumulativeDeaths)                 | `integer` | Required | cannot be null | [Covid Act Now API Schemas](schemas-definitions-actualstimeseriesrow-properties-cumulativedeaths.md "https&#x3A;//data.covidactnow.org/#/definitions/ActualsTimeseriesRow/properties/cumulativeDeaths")                 |
| [hospitalBeds](#hospitalBeds)                         | `object`  | Required | cannot be null | [Covid Act Now API Schemas](schemas-definitions-resourceutilization.md "https&#x3A;//data.covidactnow.org/#/definitions/ActualsTimeseriesRow/properties/hospitalBeds")                                                  |
| [ICUBeds](#ICUBeds)                                   | `object`  | Required | cannot be null | [Covid Act Now API Schemas](schemas-definitions-resourceutilization.md "https&#x3A;//data.covidactnow.org/#/definitions/ActualsTimeseriesRow/properties/ICUBeds")                                                       |
| [contactTracers](#contactTracers)                     | `integer` | Optional | cannot be null | [Covid Act Now API Schemas](schemas-definitions-actualstimeseriesrow-properties-contacttracers.md "https&#x3A;//data.covidactnow.org/#/definitions/ActualsTimeseriesRow/properties/contactTracers")                     |
| [date](#date)                                         | `string`  | Required | cannot be null | [Covid Act Now API Schemas](schemas-definitions-actualstimeseriesrow-properties-date.md "https&#x3A;//data.covidactnow.org/#/definitions/ActualsTimeseriesRow/properties/date")                                         |

## population

Total population in geographic region [*deprecated*: refer to summary for this]


`population`

-   is required
-   Type: `integer` ([Population](schemas-definitions-actualstimeseriesrow-properties-population.md))
-   cannot be null
-   defined in: [Covid Act Now API Schemas](schemas-definitions-actualstimeseriesrow-properties-population.md "https&#x3A;//data.covidactnow.org/#/definitions/ActualsTimeseriesRow/properties/population")

### population Type

`integer` ([Population](schemas-definitions-actualstimeseriesrow-properties-population.md))

### population Constraints

**minimum (exclusive)**: the value of this number must be greater than: `0`

## intervention

Name of high-level intervention in-place


`intervention`

-   is required
-   Type: `string` ([Intervention](schemas-definitions-actualstimeseriesrow-properties-intervention.md))
-   cannot be null
-   defined in: [Covid Act Now API Schemas](schemas-definitions-actualstimeseriesrow-properties-intervention.md "https&#x3A;//data.covidactnow.org/#/definitions/ActualsTimeseriesRow/properties/intervention")

### intervention Type

`string` ([Intervention](schemas-definitions-actualstimeseriesrow-properties-intervention.md))

## cumulativeConfirmedCases

Number of confirmed cases so far


`cumulativeConfirmedCases`

-   is required
-   Type: `integer` ([Cumulativeconfirmedcases](schemas-definitions-actualstimeseriesrow-properties-cumulativeconfirmedcases.md))
-   cannot be null
-   defined in: [Covid Act Now API Schemas](schemas-definitions-actualstimeseriesrow-properties-cumulativeconfirmedcases.md "https&#x3A;//data.covidactnow.org/#/definitions/ActualsTimeseriesRow/properties/cumulativeConfirmedCases")

### cumulativeConfirmedCases Type

`integer` ([Cumulativeconfirmedcases](schemas-definitions-actualstimeseriesrow-properties-cumulativeconfirmedcases.md))

## cumulativePositiveTests

Number of positive test results to date


`cumulativePositiveTests`

-   is required
-   Type: `integer` ([Cumulativepositivetests](schemas-definitions-actualstimeseriesrow-properties-cumulativepositivetests.md))
-   cannot be null
-   defined in: [Covid Act Now API Schemas](schemas-definitions-actualstimeseriesrow-properties-cumulativepositivetests.md "https&#x3A;//data.covidactnow.org/#/definitions/ActualsTimeseriesRow/properties/cumulativePositiveTests")

### cumulativePositiveTests Type

`integer` ([Cumulativepositivetests](schemas-definitions-actualstimeseriesrow-properties-cumulativepositivetests.md))

## cumulativeNegativeTests

Number of negative test results to date


`cumulativeNegativeTests`

-   is required
-   Type: `integer` ([Cumulativenegativetests](schemas-definitions-actualstimeseriesrow-properties-cumulativenegativetests.md))
-   cannot be null
-   defined in: [Covid Act Now API Schemas](schemas-definitions-actualstimeseriesrow-properties-cumulativenegativetests.md "https&#x3A;//data.covidactnow.org/#/definitions/ActualsTimeseriesRow/properties/cumulativeNegativeTests")

### cumulativeNegativeTests Type

`integer` ([Cumulativenegativetests](schemas-definitions-actualstimeseriesrow-properties-cumulativenegativetests.md))

## cumulativeDeaths

Number of deaths so far


`cumulativeDeaths`

-   is required
-   Type: `integer` ([Cumulativedeaths](schemas-definitions-actualstimeseriesrow-properties-cumulativedeaths.md))
-   cannot be null
-   defined in: [Covid Act Now API Schemas](schemas-definitions-actualstimeseriesrow-properties-cumulativedeaths.md "https&#x3A;//data.covidactnow.org/#/definitions/ActualsTimeseriesRow/properties/cumulativeDeaths")

### cumulativeDeaths Type

`integer` ([Cumulativedeaths](schemas-definitions-actualstimeseriesrow-properties-cumulativedeaths.md))

## hospitalBeds

Base model for API output.


`hospitalBeds`

-   is required
-   Type: `object` ([ResourceUtilization](schemas-definitions-resourceutilization.md))
-   cannot be null
-   defined in: [Covid Act Now API Schemas](schemas-definitions-resourceutilization.md "https&#x3A;//data.covidactnow.org/#/definitions/ActualsTimeseriesRow/properties/hospitalBeds")

### hospitalBeds Type

`object` ([ResourceUtilization](schemas-definitions-resourceutilization.md))

## ICUBeds

Base model for API output.


`ICUBeds`

-   is required
-   Type: `object` ([ResourceUtilization](schemas-definitions-resourceutilization.md))
-   cannot be null
-   defined in: [Covid Act Now API Schemas](schemas-definitions-resourceutilization.md "https&#x3A;//data.covidactnow.org/#/definitions/ActualsTimeseriesRow/properties/ICUBeds")

### ICUBeds Type

`object` ([ResourceUtilization](schemas-definitions-resourceutilization.md))

## contactTracers

# of Contact Tracers


`contactTracers`

-   is optional
-   Type: `integer` ([Contacttracers](schemas-definitions-actualstimeseriesrow-properties-contacttracers.md))
-   cannot be null
-   defined in: [Covid Act Now API Schemas](schemas-definitions-actualstimeseriesrow-properties-contacttracers.md "https&#x3A;//data.covidactnow.org/#/definitions/ActualsTimeseriesRow/properties/contactTracers")

### contactTracers Type

`integer` ([Contacttracers](schemas-definitions-actualstimeseriesrow-properties-contacttracers.md))

## date




`date`

-   is required
-   Type: `string` ([Date](schemas-definitions-actualstimeseriesrow-properties-date.md))
-   cannot be null
-   defined in: [Covid Act Now API Schemas](schemas-definitions-actualstimeseriesrow-properties-date.md "https&#x3A;//data.covidactnow.org/#/definitions/ActualsTimeseriesRow/properties/date")

### date Type

`string` ([Date](schemas-definitions-actualstimeseriesrow-properties-date.md))

### date Constraints

**date**: the string must be a date string, according to [RFC 3339, section 5.6](https://tools.ietf.org/html/rfc3339 "check the specification")
