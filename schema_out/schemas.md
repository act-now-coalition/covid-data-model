# Covid Act Now API Schemas Schema

```txt
https://data.covidactnow.org/
```




| Abstract               | Extensible | Status         | Identifiable            | Custom Properties | Additional Properties | Access Restrictions | Defined In                                                 |
| :--------------------- | ---------- | -------------- | ----------------------- | :---------------- | --------------------- | ------------------- | ---------------------------------------------------------- |
| Cannot be instantiated | Yes        | Unknown status | Unknown identifiability | Forbidden         | Allowed               | none                | [schemas.json](../out/schemas.json "open original schema") |

## Covid Act Now API Schemas Type

unknown ([Covid Act Now API Schemas](schemas.md))

# Covid Act Now API Schemas Definitions

## Definitions group ResourceUsageProjection

Reference this group by using

```json
{"$ref":"https://data.covidactnow.org/#/definitions/ResourceUsageProjection"}
```

Base model for API output.


`ResourceUsageProjection`

-   is optional
-   Type: `object` ([ResourceUsageProjection](schemas-definitions-resourceusageprojection.md))
-   cannot be null
-   defined in: [Covid Act Now API Schemas](schemas-definitions-resourceusageprojection.md "https&#x3A;//data.covidactnow.org/#/definitions/ResourceUsageProjection")

### ResourceUsageProjection Type

`object` ([ResourceUsageProjection](schemas-definitions-resourceusageprojection.md))

## Definitions group Projections

Reference this group by using

```json
{"$ref":"https://data.covidactnow.org/#/definitions/Projections"}
```

Base model for API output.


`Projections`

-   is optional
-   Type: `object` ([Projections](schemas-definitions-projections.md))
-   cannot be null
-   defined in: [Covid Act Now API Schemas](schemas-definitions-projections.md "https&#x3A;//data.covidactnow.org/#/definitions/Projections")

### Projections Type

`object` ([Projections](schemas-definitions-projections.md))

## Definitions group ResourceUtilization

Reference this group by using

```json
{"$ref":"https://data.covidactnow.org/#/definitions/ResourceUtilization"}
```

Base model for API output.


`ResourceUtilization`

-   is optional
-   Type: `object` ([ResourceUtilization](schemas-definitions-resourceutilization.md))
-   cannot be null
-   defined in: [Covid Act Now API Schemas](schemas-definitions-resourceutilization.md "https&#x3A;//data.covidactnow.org/#/definitions/ResourceUtilization")

### ResourceUtilization Type

`object` ([ResourceUtilization](schemas-definitions-resourceutilization.md))

## Definitions group Actuals

Reference this group by using

```json
{"$ref":"https://data.covidactnow.org/#/definitions/Actuals"}
```

Base model for API output.


`Actuals`

-   is optional
-   Type: `object` ([Actuals](schemas-definitions-actuals.md))
-   cannot be null
-   defined in: [Covid Act Now API Schemas](schemas-definitions-actuals.md "https&#x3A;//data.covidactnow.org/#/definitions/Actuals")

### Actuals Type

`object` ([Actuals](schemas-definitions-actuals.md))

## Definitions group ActualsTimeseriesRow

Reference this group by using

```json
{"$ref":"https://data.covidactnow.org/#/definitions/ActualsTimeseriesRow"}
```

Base model for API output.


`ActualsTimeseriesRow`

-   is optional
-   Type: `object` ([ActualsTimeseriesRow](schemas-definitions-actualstimeseriesrow.md))
-   cannot be null
-   defined in: [Covid Act Now API Schemas](schemas-definitions-actualstimeseriesrow.md "https&#x3A;//data.covidactnow.org/#/definitions/ActualsTimeseriesRow")

### ActualsTimeseriesRow Type

`object` ([ActualsTimeseriesRow](schemas-definitions-actualstimeseriesrow.md))

## Definitions group RegionSummary

Reference this group by using

```json
{"$ref":"https://data.covidactnow.org/#/definitions/RegionSummary"}
```

Base model for API output.


`RegionSummary`

-   is optional
-   Type: `object` ([RegionSummary](schemas-definitions-regionsummary.md))
-   cannot be null
-   defined in: [Covid Act Now API Schemas](schemas-definitions-regionsummary.md "https&#x3A;//data.covidactnow.org/#/definitions/RegionSummary")

### RegionSummary Type

`object` ([RegionSummary](schemas-definitions-regionsummary.md))

## Definitions group PredictionTimeseriesRow

Reference this group by using

```json
{"$ref":"https://data.covidactnow.org/#/definitions/PredictionTimeseriesRow"}
```

Base model for API output.


`PredictionTimeseriesRow`

-   is optional
-   Type: `object` ([PredictionTimeseriesRow](schemas-definitions-predictiontimeseriesrow.md))
-   cannot be null
-   defined in: [Covid Act Now API Schemas](schemas-definitions-predictiontimeseriesrow.md "https&#x3A;//data.covidactnow.org/#/definitions/PredictionTimeseriesRow")

### PredictionTimeseriesRow Type

`object` ([PredictionTimeseriesRow](schemas-definitions-predictiontimeseriesrow.md))

## Definitions group RegionSummaryWithTimeseries

Reference this group by using

```json
{"$ref":"https://data.covidactnow.org/#/definitions/RegionSummaryWithTimeseries"}
```

Base model for API output.


`RegionSummaryWithTimeseries`

-   is optional
-   Type: `object` ([RegionSummaryWithTimeseries](schemas-definitions-regionsummarywithtimeseries.md))
-   cannot be null
-   defined in: [Covid Act Now API Schemas](schemas-definitions-regionsummarywithtimeseries.md "https&#x3A;//data.covidactnow.org/#/definitions/RegionSummaryWithTimeseries")

### RegionSummaryWithTimeseries Type

`object` ([RegionSummaryWithTimeseries](schemas-definitions-regionsummarywithtimeseries.md))

## Definitions group PredictionTimeseriesRowWithHeader

Reference this group by using

```json
{"$ref":"https://data.covidactnow.org/#/definitions/PredictionTimeseriesRowWithHeader"}
```

Base model for API output.


`PredictionTimeseriesRowWithHeader`

-   is optional
-   Type: `object` ([PredictionTimeseriesRowWithHeader](schemas-definitions-predictiontimeseriesrowwithheader.md))
-   cannot be null
-   defined in: [Covid Act Now API Schemas](schemas-definitions-predictiontimeseriesrowwithheader.md "https&#x3A;//data.covidactnow.org/#/definitions/PredictionTimeseriesRowWithHeader")

### PredictionTimeseriesRowWithHeader Type

`object` ([PredictionTimeseriesRowWithHeader](schemas-definitions-predictiontimeseriesrowwithheader.md))

## Definitions group AggregateRegionSummary

Reference this group by using

```json
{"$ref":"https://data.covidactnow.org/#/definitions/AggregateRegionSummary"}
```

Base model for API output.


`AggregateRegionSummary`

-   is optional
-   Type: `object[]` ([RegionSummary](schemas-definitions-regionsummary.md))
-   cannot be null
-   defined in: [Covid Act Now API Schemas](schemas-definitions-aggregateregionsummary.md "https&#x3A;//data.covidactnow.org/#/definitions/AggregateRegionSummary")

### AggregateRegionSummary Type

`object[]` ([RegionSummary](schemas-definitions-regionsummary.md))

## Definitions group AggregateRegionSummaryWithTimeseries

Reference this group by using

```json
{"$ref":"https://data.covidactnow.org/#/definitions/AggregateRegionSummaryWithTimeseries"}
```

Base model for API output.


`AggregateRegionSummaryWithTimeseries`

-   is optional
-   Type: `object[]` ([RegionSummaryWithTimeseries](schemas-definitions-regionsummarywithtimeseries.md))
-   cannot be null
-   defined in: [Covid Act Now API Schemas](schemas-definitions-aggregateregionsummarywithtimeseries.md "https&#x3A;//data.covidactnow.org/#/definitions/AggregateRegionSummaryWithTimeseries")

### AggregateRegionSummaryWithTimeseries Type

`object[]` ([RegionSummaryWithTimeseries](schemas-definitions-regionsummarywithtimeseries.md))

## Definitions group AggregateFlattenedTimeseries

Reference this group by using

```json
{"$ref":"https://data.covidactnow.org/#/definitions/AggregateFlattenedTimeseries"}
```

Base model for API output.


`AggregateFlattenedTimeseries`

-   is optional
-   Type: `object[]` ([PredictionTimeseriesRowWithHeader](schemas-definitions-predictiontimeseriesrowwithheader.md))
-   cannot be null
-   defined in: [Covid Act Now API Schemas](schemas-definitions-aggregateflattenedtimeseries.md "https&#x3A;//data.covidactnow.org/#/definitions/AggregateFlattenedTimeseries")

### AggregateFlattenedTimeseries Type

`object[]` ([PredictionTimeseriesRowWithHeader](schemas-definitions-predictiontimeseriesrowwithheader.md))
