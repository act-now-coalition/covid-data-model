# Actuals Schema

```txt
https://data.covidactnow.org/#/definitions/RegionSummaryWithTimeseries/properties/actuals
```

Base model for API output.


| Abstract            | Extensible | Status         | Identifiable | Custom Properties | Additional Properties | Access Restrictions | Defined In                                                   |
| :------------------ | ---------- | -------------- | ------------ | :---------------- | --------------------- | ------------------- | ------------------------------------------------------------ |
| Can be instantiated | No         | Unknown status | No           | Forbidden         | Allowed               | none                | [schemas.json\*](../out/schemas.json "open original schema") |

## actuals Type

`object` ([Actuals](schemas-definitions-actuals.md))

# Actuals Properties

| Property                                              | Type      | Required | Nullable       | Defined by                                                                                                                                                                                    |
| :---------------------------------------------------- | --------- | -------- | -------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [population](#population)                             | `integer` | Required | cannot be null | [Covid Act Now API Schemas](schemas-definitions-actuals-properties-population.md "https&#x3A;//data.covidactnow.org/#/definitions/Actuals/properties/population")                             |
| [intervention](#intervention)                         | `string`  | Required | cannot be null | [Covid Act Now API Schemas](schemas-definitions-actuals-properties-intervention.md "https&#x3A;//data.covidactnow.org/#/definitions/Actuals/properties/intervention")                         |
| [cumulativeConfirmedCases](#cumulativeConfirmedCases) | `integer` | Required | cannot be null | [Covid Act Now API Schemas](schemas-definitions-actuals-properties-cumulativeconfirmedcases.md "https&#x3A;//data.covidactnow.org/#/definitions/Actuals/properties/cumulativeConfirmedCases") |
| [cumulativePositiveTests](#cumulativePositiveTests)   | `integer` | Required | cannot be null | [Covid Act Now API Schemas](schemas-definitions-actuals-properties-cumulativepositivetests.md "https&#x3A;//data.covidactnow.org/#/definitions/Actuals/properties/cumulativePositiveTests")   |
| [cumulativeNegativeTests](#cumulativeNegativeTests)   | `integer` | Required | cannot be null | [Covid Act Now API Schemas](schemas-definitions-actuals-properties-cumulativenegativetests.md "https&#x3A;//data.covidactnow.org/#/definitions/Actuals/properties/cumulativeNegativeTests")   |
| [cumulativeDeaths](#cumulativeDeaths)                 | `integer` | Required | cannot be null | [Covid Act Now API Schemas](schemas-definitions-actuals-properties-cumulativedeaths.md "https&#x3A;//data.covidactnow.org/#/definitions/Actuals/properties/cumulativeDeaths")                 |
| [hospitalBeds](#hospitalBeds)                         | `object`  | Required | cannot be null | [Covid Act Now API Schemas](schemas-definitions-resourceutilization.md "https&#x3A;//data.covidactnow.org/#/definitions/Actuals/properties/hospitalBeds")                                     |
| [ICUBeds](#ICUBeds)                                   | `object`  | Required | cannot be null | [Covid Act Now API Schemas](schemas-definitions-resourceutilization.md "https&#x3A;//data.covidactnow.org/#/definitions/Actuals/properties/ICUBeds")                                          |
| [contactTracers](#contactTracers)                     | `integer` | Optional | cannot be null | [Covid Act Now API Schemas](schemas-definitions-actuals-properties-contacttracers.md "https&#x3A;//data.covidactnow.org/#/definitions/Actuals/properties/contactTracers")                     |

## population

Total population in geographic region [*deprecated*: refer to summary for this]


`population`

-   is required
-   Type: `integer` ([Population](schemas-definitions-actuals-properties-population.md))
-   cannot be null
-   defined in: [Covid Act Now API Schemas](schemas-definitions-actuals-properties-population.md "https&#x3A;//data.covidactnow.org/#/definitions/Actuals/properties/population")

### population Type

`integer` ([Population](schemas-definitions-actuals-properties-population.md))

### population Constraints

**minimum (exclusive)**: the value of this number must be greater than: `0`

## intervention

Name of high-level intervention in-place


`intervention`

-   is required
-   Type: `string` ([Intervention](schemas-definitions-actuals-properties-intervention.md))
-   cannot be null
-   defined in: [Covid Act Now API Schemas](schemas-definitions-actuals-properties-intervention.md "https&#x3A;//data.covidactnow.org/#/definitions/Actuals/properties/intervention")

### intervention Type

`string` ([Intervention](schemas-definitions-actuals-properties-intervention.md))

## cumulativeConfirmedCases

Number of confirmed cases so far


`cumulativeConfirmedCases`

-   is required
-   Type: `integer` ([Cumulativeconfirmedcases](schemas-definitions-actuals-properties-cumulativeconfirmedcases.md))
-   cannot be null
-   defined in: [Covid Act Now API Schemas](schemas-definitions-actuals-properties-cumulativeconfirmedcases.md "https&#x3A;//data.covidactnow.org/#/definitions/Actuals/properties/cumulativeConfirmedCases")

### cumulativeConfirmedCases Type

`integer` ([Cumulativeconfirmedcases](schemas-definitions-actuals-properties-cumulativeconfirmedcases.md))

## cumulativePositiveTests

Number of positive test results to date


`cumulativePositiveTests`

-   is required
-   Type: `integer` ([Cumulativepositivetests](schemas-definitions-actuals-properties-cumulativepositivetests.md))
-   cannot be null
-   defined in: [Covid Act Now API Schemas](schemas-definitions-actuals-properties-cumulativepositivetests.md "https&#x3A;//data.covidactnow.org/#/definitions/Actuals/properties/cumulativePositiveTests")

### cumulativePositiveTests Type

`integer` ([Cumulativepositivetests](schemas-definitions-actuals-properties-cumulativepositivetests.md))

## cumulativeNegativeTests

Number of negative test results to date


`cumulativeNegativeTests`

-   is required
-   Type: `integer` ([Cumulativenegativetests](schemas-definitions-actuals-properties-cumulativenegativetests.md))
-   cannot be null
-   defined in: [Covid Act Now API Schemas](schemas-definitions-actuals-properties-cumulativenegativetests.md "https&#x3A;//data.covidactnow.org/#/definitions/Actuals/properties/cumulativeNegativeTests")

### cumulativeNegativeTests Type

`integer` ([Cumulativenegativetests](schemas-definitions-actuals-properties-cumulativenegativetests.md))

## cumulativeDeaths

Number of deaths so far


`cumulativeDeaths`

-   is required
-   Type: `integer` ([Cumulativedeaths](schemas-definitions-actuals-properties-cumulativedeaths.md))
-   cannot be null
-   defined in: [Covid Act Now API Schemas](schemas-definitions-actuals-properties-cumulativedeaths.md "https&#x3A;//data.covidactnow.org/#/definitions/Actuals/properties/cumulativeDeaths")

### cumulativeDeaths Type

`integer` ([Cumulativedeaths](schemas-definitions-actuals-properties-cumulativedeaths.md))

## hospitalBeds

Base model for API output.


`hospitalBeds`

-   is required
-   Type: `object` ([ResourceUtilization](schemas-definitions-resourceutilization.md))
-   cannot be null
-   defined in: [Covid Act Now API Schemas](schemas-definitions-resourceutilization.md "https&#x3A;//data.covidactnow.org/#/definitions/Actuals/properties/hospitalBeds")

### hospitalBeds Type

`object` ([ResourceUtilization](schemas-definitions-resourceutilization.md))

## ICUBeds

Base model for API output.


`ICUBeds`

-   is required
-   Type: `object` ([ResourceUtilization](schemas-definitions-resourceutilization.md))
-   cannot be null
-   defined in: [Covid Act Now API Schemas](schemas-definitions-resourceutilization.md "https&#x3A;//data.covidactnow.org/#/definitions/Actuals/properties/ICUBeds")

### ICUBeds Type

`object` ([ResourceUtilization](schemas-definitions-resourceutilization.md))

## contactTracers

# of Contact Tracers


`contactTracers`

-   is optional
-   Type: `integer` ([Contacttracers](schemas-definitions-actuals-properties-contacttracers.md))
-   cannot be null
-   defined in: [Covid Act Now API Schemas](schemas-definitions-actuals-properties-contacttracers.md "https&#x3A;//data.covidactnow.org/#/definitions/Actuals/properties/contactTracers")

### contactTracers Type

`integer` ([Contacttracers](schemas-definitions-actuals-properties-contacttracers.md))
