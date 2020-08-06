# ResourceUsageProjection Schema

```txt
https://data.covidactnow.org/#/definitions/Projections/properties/ICUBeds/allOf/0
```

Base model for API output.


| Abstract            | Extensible | Status         | Identifiable | Custom Properties | Additional Properties | Access Restrictions | Defined In                                                   |
| :------------------ | ---------- | -------------- | ------------ | :---------------- | --------------------- | ------------------- | ------------------------------------------------------------ |
| Can be instantiated | No         | Unknown status | No           | Forbidden         | Allowed               | none                | [schemas.json\*](../out/schemas.json "open original schema") |

## 0 Type

`object` ([ResourceUsageProjection](schemas-definitions-resourceusageprojection.md))

# ResourceUsageProjection Properties

| Property                                | Type      | Required | Nullable       | Defined by                                                                                                                                                                                                      |
| :-------------------------------------- | --------- | -------- | -------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [peakShortfall](#peakShortfall)         | `integer` | Required | cannot be null | [Covid Act Now API Schemas](schemas-definitions-resourceusageprojection-properties-peakshortfall.md "https&#x3A;//data.covidactnow.org/#/definitions/ResourceUsageProjection/properties/peakShortfall")         |
| [peakDate](#peakDate)                   | `string`  | Required | cannot be null | [Covid Act Now API Schemas](schemas-definitions-resourceusageprojection-properties-peakdate.md "https&#x3A;//data.covidactnow.org/#/definitions/ResourceUsageProjection/properties/peakDate")                   |
| [shortageStartDate](#shortageStartDate) | `string`  | Required | cannot be null | [Covid Act Now API Schemas](schemas-definitions-resourceusageprojection-properties-shortagestartdate.md "https&#x3A;//data.covidactnow.org/#/definitions/ResourceUsageProjection/properties/shortageStartDate") |

## peakShortfall

Shortfall of resource needed at the peak utilization


`peakShortfall`

-   is required
-   Type: `integer` ([Peakshortfall](schemas-definitions-resourceusageprojection-properties-peakshortfall.md))
-   cannot be null
-   defined in: [Covid Act Now API Schemas](schemas-definitions-resourceusageprojection-properties-peakshortfall.md "https&#x3A;//data.covidactnow.org/#/definitions/ResourceUsageProjection/properties/peakShortfall")

### peakShortfall Type

`integer` ([Peakshortfall](schemas-definitions-resourceusageprojection-properties-peakshortfall.md))

## peakDate

Date of peak resource utilization


`peakDate`

-   is required
-   Type: `string` ([Peakdate](schemas-definitions-resourceusageprojection-properties-peakdate.md))
-   cannot be null
-   defined in: [Covid Act Now API Schemas](schemas-definitions-resourceusageprojection-properties-peakdate.md "https&#x3A;//data.covidactnow.org/#/definitions/ResourceUsageProjection/properties/peakDate")

### peakDate Type

`string` ([Peakdate](schemas-definitions-resourceusageprojection-properties-peakdate.md))

### peakDate Constraints

**date**: the string must be a date string, according to [RFC 3339, section 5.6](https://tools.ietf.org/html/rfc3339 "check the specification")

## shortageStartDate

Date when resource shortage begins


`shortageStartDate`

-   is required
-   Type: `string` ([Shortagestartdate](schemas-definitions-resourceusageprojection-properties-shortagestartdate.md))
-   cannot be null
-   defined in: [Covid Act Now API Schemas](schemas-definitions-resourceusageprojection-properties-shortagestartdate.md "https&#x3A;//data.covidactnow.org/#/definitions/ResourceUsageProjection/properties/shortageStartDate")

### shortageStartDate Type

`string` ([Shortagestartdate](schemas-definitions-resourceusageprojection-properties-shortagestartdate.md))

### shortageStartDate Constraints

**date**: the string must be a date string, according to [RFC 3339, section 5.6](https://tools.ietf.org/html/rfc3339 "check the specification")
