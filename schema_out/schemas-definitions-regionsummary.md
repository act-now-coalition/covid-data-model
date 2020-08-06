# RegionSummary Schema

```txt
https://data.covidactnow.org/#/definitions/AggregateRegionSummary/items
```

Base model for API output.


| Abstract            | Extensible | Status         | Identifiable | Custom Properties | Additional Properties | Access Restrictions | Defined In                                                   |
| :------------------ | ---------- | -------------- | ------------ | :---------------- | --------------------- | ------------------- | ------------------------------------------------------------ |
| Can be instantiated | No         | Unknown status | No           | Forbidden         | Allowed               | none                | [schemas.json\*](../out/schemas.json "open original schema") |

## items Type

`object` ([RegionSummary](schemas-definitions-regionsummary.md))

# RegionSummary Properties

| Property                            | Type      | Required | Nullable       | Defined by                                                                                                                                                                              |
| :---------------------------------- | --------- | -------- | -------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [countryName](#countryName)         | `string`  | Optional | cannot be null | [Covid Act Now API Schemas](schemas-definitions-regionsummary-properties-countryname.md "https&#x3A;//data.covidactnow.org/#/definitions/RegionSummary/properties/countryName")         |
| [fips](#fips)                       | `string`  | Required | cannot be null | [Covid Act Now API Schemas](schemas-definitions-regionsummary-properties-fips.md "https&#x3A;//data.covidactnow.org/#/definitions/RegionSummary/properties/fips")                       |
| [lat](#lat)                         | `number`  | Required | cannot be null | [Covid Act Now API Schemas](schemas-definitions-regionsummary-properties-lat.md "https&#x3A;//data.covidactnow.org/#/definitions/RegionSummary/properties/lat")                         |
| [long](#long)                       | `number`  | Required | cannot be null | [Covid Act Now API Schemas](schemas-definitions-regionsummary-properties-long.md "https&#x3A;//data.covidactnow.org/#/definitions/RegionSummary/properties/long")                       |
| [stateName](#stateName)             | `string`  | Required | cannot be null | [Covid Act Now API Schemas](schemas-definitions-regionsummary-properties-statename.md "https&#x3A;//data.covidactnow.org/#/definitions/RegionSummary/properties/stateName")             |
| [countyName](#countyName)           | `string`  | Optional | cannot be null | [Covid Act Now API Schemas](schemas-definitions-regionsummary-properties-countyname.md "https&#x3A;//data.covidactnow.org/#/definitions/RegionSummary/properties/countyName")           |
| [lastUpdatedDate](#lastUpdatedDate) | `string`  | Required | cannot be null | [Covid Act Now API Schemas](schemas-definitions-regionsummary-properties-lastupdateddate.md "https&#x3A;//data.covidactnow.org/#/definitions/RegionSummary/properties/lastUpdatedDate") |
| [projections](#projections)         | `object`  | Required | cannot be null | [Covid Act Now API Schemas](schemas-definitions-projections.md "https&#x3A;//data.covidactnow.org/#/definitions/RegionSummary/properties/projections")                                  |
| [actuals](#actuals)                 | `object`  | Required | cannot be null | [Covid Act Now API Schemas](schemas-definitions-actuals.md "https&#x3A;//data.covidactnow.org/#/definitions/RegionSummary/properties/actuals")                                          |
| [population](#population)           | `integer` | Required | cannot be null | [Covid Act Now API Schemas](schemas-definitions-regionsummary-properties-population.md "https&#x3A;//data.covidactnow.org/#/definitions/RegionSummary/properties/population")           |

## countryName




`countryName`

-   is optional
-   Type: `string` ([Countryname](schemas-definitions-regionsummary-properties-countryname.md))
-   cannot be null
-   defined in: [Covid Act Now API Schemas](schemas-definitions-regionsummary-properties-countryname.md "https&#x3A;//data.covidactnow.org/#/definitions/RegionSummary/properties/countryName")

### countryName Type

`string` ([Countryname](schemas-definitions-regionsummary-properties-countryname.md))

### countryName Default Value

The default value is:

```json
"US"
```

## fips

Fips Code.  For state level data, 2 characters, for county level data, 5 characters.


`fips`

-   is required
-   Type: `string` ([Fips](schemas-definitions-regionsummary-properties-fips.md))
-   cannot be null
-   defined in: [Covid Act Now API Schemas](schemas-definitions-regionsummary-properties-fips.md "https&#x3A;//data.covidactnow.org/#/definitions/RegionSummary/properties/fips")

### fips Type

`string` ([Fips](schemas-definitions-regionsummary-properties-fips.md))

## lat

Latitude of point within the state or county


`lat`

-   is required
-   Type: `number` ([Lat](schemas-definitions-regionsummary-properties-lat.md))
-   cannot be null
-   defined in: [Covid Act Now API Schemas](schemas-definitions-regionsummary-properties-lat.md "https&#x3A;//data.covidactnow.org/#/definitions/RegionSummary/properties/lat")

### lat Type

`number` ([Lat](schemas-definitions-regionsummary-properties-lat.md))

## long

Longitude of point within the state or county


`long`

-   is required
-   Type: `number` ([Long](schemas-definitions-regionsummary-properties-long.md))
-   cannot be null
-   defined in: [Covid Act Now API Schemas](schemas-definitions-regionsummary-properties-long.md "https&#x3A;//data.covidactnow.org/#/definitions/RegionSummary/properties/long")

### long Type

`number` ([Long](schemas-definitions-regionsummary-properties-long.md))

## stateName

The state name


`stateName`

-   is required
-   Type: `string` ([Statename](schemas-definitions-regionsummary-properties-statename.md))
-   cannot be null
-   defined in: [Covid Act Now API Schemas](schemas-definitions-regionsummary-properties-statename.md "https&#x3A;//data.covidactnow.org/#/definitions/RegionSummary/properties/stateName")

### stateName Type

`string` ([Statename](schemas-definitions-regionsummary-properties-statename.md))

## countyName

The county name


`countyName`

-   is optional
-   Type: `string` ([Countyname](schemas-definitions-regionsummary-properties-countyname.md))
-   cannot be null
-   defined in: [Covid Act Now API Schemas](schemas-definitions-regionsummary-properties-countyname.md "https&#x3A;//data.covidactnow.org/#/definitions/RegionSummary/properties/countyName")

### countyName Type

`string` ([Countyname](schemas-definitions-regionsummary-properties-countyname.md))

## lastUpdatedDate

Date of latest data


`lastUpdatedDate`

-   is required
-   Type: `string` ([Lastupdateddate](schemas-definitions-regionsummary-properties-lastupdateddate.md))
-   cannot be null
-   defined in: [Covid Act Now API Schemas](schemas-definitions-regionsummary-properties-lastupdateddate.md "https&#x3A;//data.covidactnow.org/#/definitions/RegionSummary/properties/lastUpdatedDate")

### lastUpdatedDate Type

`string` ([Lastupdateddate](schemas-definitions-regionsummary-properties-lastupdateddate.md))

### lastUpdatedDate Constraints

**date**: the string must be a date string, according to [RFC 3339, section 5.6](https://tools.ietf.org/html/rfc3339 "check the specification")

## projections

Base model for API output.


`projections`

-   is required
-   Type: `object` ([Projections](schemas-definitions-projections.md))
-   cannot be null
-   defined in: [Covid Act Now API Schemas](schemas-definitions-projections.md "https&#x3A;//data.covidactnow.org/#/definitions/RegionSummary/properties/projections")

### projections Type

`object` ([Projections](schemas-definitions-projections.md))

## actuals

Base model for API output.


`actuals`

-   is required
-   Type: `object` ([Actuals](schemas-definitions-actuals.md))
-   cannot be null
-   defined in: [Covid Act Now API Schemas](schemas-definitions-actuals.md "https&#x3A;//data.covidactnow.org/#/definitions/RegionSummary/properties/actuals")

### actuals Type

`object` ([Actuals](schemas-definitions-actuals.md))

## population

Total Population in geographic region.


`population`

-   is required
-   Type: `integer` ([Population](schemas-definitions-regionsummary-properties-population.md))
-   cannot be null
-   defined in: [Covid Act Now API Schemas](schemas-definitions-regionsummary-properties-population.md "https&#x3A;//data.covidactnow.org/#/definitions/RegionSummary/properties/population")

### population Type

`integer` ([Population](schemas-definitions-regionsummary-properties-population.md))

### population Constraints

**minimum (exclusive)**: the value of this number must be greater than: `0`
