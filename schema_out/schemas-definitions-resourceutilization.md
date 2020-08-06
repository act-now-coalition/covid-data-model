# ResourceUtilization Schema

```txt
https://data.covidactnow.org/#/definitions/ActualsTimeseriesRow/properties/ICUBeds
```

Base model for API output.


| Abstract            | Extensible | Status         | Identifiable | Custom Properties | Additional Properties | Access Restrictions | Defined In                                                   |
| :------------------ | ---------- | -------------- | ------------ | :---------------- | --------------------- | ------------------- | ------------------------------------------------------------ |
| Can be instantiated | No         | Unknown status | No           | Forbidden         | Allowed               | none                | [schemas.json\*](../out/schemas.json "open original schema") |

## ICUBeds Type

`object` ([ResourceUtilization](schemas-definitions-resourceutilization.md))

# ResourceUtilization Properties

| Property                                | Type      | Required | Nullable       | Defined by                                                                                                                                                                                              |
| :-------------------------------------- | --------- | -------- | -------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| [capacity](#capacity)                   | `integer` | Required | cannot be null | [Covid Act Now API Schemas](schemas-definitions-resourceutilization-properties-capacity.md "https&#x3A;//data.covidactnow.org/#/definitions/ResourceUtilization/properties/capacity")                   |
| [totalCapacity](#totalCapacity)         | `integer` | Required | cannot be null | [Covid Act Now API Schemas](schemas-definitions-resourceutilization-properties-totalcapacity.md "https&#x3A;//data.covidactnow.org/#/definitions/ResourceUtilization/properties/totalCapacity")         |
| [currentUsageCovid](#currentUsageCovid) | `integer` | Required | cannot be null | [Covid Act Now API Schemas](schemas-definitions-resourceutilization-properties-currentusagecovid.md "https&#x3A;//data.covidactnow.org/#/definitions/ResourceUtilization/properties/currentUsageCovid") |
| [currentUsageTotal](#currentUsageTotal) | `integer` | Required | cannot be null | [Covid Act Now API Schemas](schemas-definitions-resourceutilization-properties-currentusagetotal.md "https&#x3A;//data.covidactnow.org/#/definitions/ResourceUtilization/properties/currentUsageTotal") |
| [typicalUsageRate](#typicalUsageRate)   | `number`  | Required | cannot be null | [Covid Act Now API Schemas](schemas-definitions-resourceutilization-properties-typicalusagerate.md "https&#x3A;//data.covidactnow.org/#/definitions/ResourceUtilization/properties/typicalUsageRate")   |

## capacity

_deprecated_: Capacity for resource. In the case of ICUs, this refers to total capacity. For hospitalization this refers to free capacity for COVID patients. This value is calculated by (1 - typicalUsageRate) _ totalCapacity _ 2.07


`capacity`

-   is required
-   Type: `integer` ([Capacity](schemas-definitions-resourceutilization-properties-capacity.md))
-   cannot be null
-   defined in: [Covid Act Now API Schemas](schemas-definitions-resourceutilization-properties-capacity.md "https&#x3A;//data.covidactnow.org/#/definitions/ResourceUtilization/properties/capacity")

### capacity Type

`integer` ([Capacity](schemas-definitions-resourceutilization-properties-capacity.md))

## totalCapacity

Total capacity for resource.


`totalCapacity`

-   is required
-   Type: `integer` ([Totalcapacity](schemas-definitions-resourceutilization-properties-totalcapacity.md))
-   cannot be null
-   defined in: [Covid Act Now API Schemas](schemas-definitions-resourceutilization-properties-totalcapacity.md "https&#x3A;//data.covidactnow.org/#/definitions/ResourceUtilization/properties/totalCapacity")

### totalCapacity Type

`integer` ([Totalcapacity](schemas-definitions-resourceutilization-properties-totalcapacity.md))

## currentUsageCovid

Currently used capacity for resource by COVID 


`currentUsageCovid`

-   is required
-   Type: `integer` ([Currentusagecovid](schemas-definitions-resourceutilization-properties-currentusagecovid.md))
-   cannot be null
-   defined in: [Covid Act Now API Schemas](schemas-definitions-resourceutilization-properties-currentusagecovid.md "https&#x3A;//data.covidactnow.org/#/definitions/ResourceUtilization/properties/currentUsageCovid")

### currentUsageCovid Type

`integer` ([Currentusagecovid](schemas-definitions-resourceutilization-properties-currentusagecovid.md))

## currentUsageTotal

Currently used capacity for resource by all patients (COVID + Non-COVID)


`currentUsageTotal`

-   is required
-   Type: `integer` ([Currentusagetotal](schemas-definitions-resourceutilization-properties-currentusagetotal.md))
-   cannot be null
-   defined in: [Covid Act Now API Schemas](schemas-definitions-resourceutilization-properties-currentusagetotal.md "https&#x3A;//data.covidactnow.org/#/definitions/ResourceUtilization/properties/currentUsageTotal")

### currentUsageTotal Type

`integer` ([Currentusagetotal](schemas-definitions-resourceutilization-properties-currentusagetotal.md))

## typicalUsageRate

Typical used capacity rate for resource. This excludes any COVID usage.


`typicalUsageRate`

-   is required
-   Type: `number` ([Typicalusagerate](schemas-definitions-resourceutilization-properties-typicalusagerate.md))
-   cannot be null
-   defined in: [Covid Act Now API Schemas](schemas-definitions-resourceutilization-properties-typicalusagerate.md "https&#x3A;//data.covidactnow.org/#/definitions/ResourceUtilization/properties/typicalUsageRate")

### typicalUsageRate Type

`number` ([Typicalusagerate](schemas-definitions-resourceutilization-properties-typicalusagerate.md))
