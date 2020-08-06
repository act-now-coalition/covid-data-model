# Capacity Schema

```txt
https://data.covidactnow.org/#/definitions/ResourceUtilization/properties/capacity
```

_deprecated_: Capacity for resource. In the case of ICUs, this refers to total capacity. For hospitalization this refers to free capacity for COVID patients. This value is calculated by (1 - typicalUsageRate) _ totalCapacity _ 2.07


| Abstract            | Extensible | Status         | Identifiable            | Custom Properties | Additional Properties | Access Restrictions | Defined In                                                   |
| :------------------ | ---------- | -------------- | ----------------------- | :---------------- | --------------------- | ------------------- | ------------------------------------------------------------ |
| Can be instantiated | No         | Unknown status | Unknown identifiability | Forbidden         | Allowed               | none                | [schemas.json\*](../out/schemas.json "open original schema") |

## capacity Type

`integer` ([Capacity](schemas-definitions-resourceutilization-properties-capacity.md))
