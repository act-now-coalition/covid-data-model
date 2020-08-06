# Projections Schema

```txt
https://data.covidactnow.org/#/definitions/RegionSummaryWithTimeseries/properties/projections
```

Base model for API output.


| Abstract            | Extensible | Status         | Identifiable | Custom Properties | Additional Properties | Access Restrictions | Defined In                                                   |
| :------------------ | ---------- | -------------- | ------------ | :---------------- | --------------------- | ------------------- | ------------------------------------------------------------ |
| Can be instantiated | No         | Unknown status | No           | Forbidden         | Allowed               | none                | [schemas.json\*](../out/schemas.json "open original schema") |

## projections Type

`object` ([Projections](schemas-definitions-projections.md))

# Projections Properties

| Property                                | Type     | Required | Nullable       | Defined by                                                                                                                                                                              |
| :-------------------------------------- | -------- | -------- | -------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [totalHospitalBeds](#totalHospitalBeds) | Merged   | Required | cannot be null | [Covid Act Now API Schemas](schemas-definitions-projections-properties-totalhospitalbeds.md "https&#x3A;//data.covidactnow.org/#/definitions/Projections/properties/totalHospitalBeds") |
| [ICUBeds](#ICUBeds)                     | Merged   | Required | cannot be null | [Covid Act Now API Schemas](schemas-definitions-projections-properties-icubeds.md "https&#x3A;//data.covidactnow.org/#/definitions/Projections/properties/ICUBeds")                     |
| [Rt](#Rt)                               | `number` | Required | cannot be null | [Covid Act Now API Schemas](schemas-definitions-projections-properties-rt.md "https&#x3A;//data.covidactnow.org/#/definitions/Projections/properties/Rt")                               |
| [RtCI90](#RtCI90)                       | `number` | Required | cannot be null | [Covid Act Now API Schemas](schemas-definitions-projections-properties-rtci90.md "https&#x3A;//data.covidactnow.org/#/definitions/Projections/properties/RtCI90")                       |

## totalHospitalBeds

Projection about total hospital bed utilization


`totalHospitalBeds`

-   is required
-   Type: merged type ([Totalhospitalbeds](schemas-definitions-projections-properties-totalhospitalbeds.md))
-   cannot be null
-   defined in: [Covid Act Now API Schemas](schemas-definitions-projections-properties-totalhospitalbeds.md "https&#x3A;//data.covidactnow.org/#/definitions/Projections/properties/totalHospitalBeds")

### totalHospitalBeds Type

merged type ([Totalhospitalbeds](schemas-definitions-projections-properties-totalhospitalbeds.md))

all of

-   [ResourceUsageProjection](schemas-definitions-resourceusageprojection.md "check type definition")

## ICUBeds

Projection about ICU hospital bed utilization


`ICUBeds`

-   is required
-   Type: merged type ([Icubeds](schemas-definitions-projections-properties-icubeds.md))
-   cannot be null
-   defined in: [Covid Act Now API Schemas](schemas-definitions-projections-properties-icubeds.md "https&#x3A;//data.covidactnow.org/#/definitions/Projections/properties/ICUBeds")

### ICUBeds Type

merged type ([Icubeds](schemas-definitions-projections-properties-icubeds.md))

all of

-   [ResourceUsageProjection](schemas-definitions-resourceusageprojection.md "check type definition")

## Rt

Inferred Rt


`Rt`

-   is required
-   Type: `number` ([Rt](schemas-definitions-projections-properties-rt.md))
-   cannot be null
-   defined in: [Covid Act Now API Schemas](schemas-definitions-projections-properties-rt.md "https&#x3A;//data.covidactnow.org/#/definitions/Projections/properties/Rt")

### Rt Type

`number` ([Rt](schemas-definitions-projections-properties-rt.md))

## RtCI90

Rt 90th percentile confidence interval upper endpoint.


`RtCI90`

-   is required
-   Type: `number` ([Rtci90](schemas-definitions-projections-properties-rtci90.md))
-   cannot be null
-   defined in: [Covid Act Now API Schemas](schemas-definitions-projections-properties-rtci90.md "https&#x3A;//data.covidactnow.org/#/definitions/Projections/properties/RtCI90")

### RtCI90 Type

`number` ([Rtci90](schemas-definitions-projections-properties-rtci90.md))
