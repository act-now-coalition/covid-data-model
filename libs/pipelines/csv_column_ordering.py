"""
Module to specify columns in output csvs.

Changing the column ordering breaks many of our users who rely on column indices
to reference in excel and google sheets.  The timeseries and summary csvs use these
to determine column ordering.  Recommended to be append only.
"""

SUMMARY_ORDER = [
    "fips",
    "country",
    "state",
    "county",
    "level",
    "lat",
    "locationId",
    "long",
    "population",
    "metrics.testPositivityRatio",
    "metrics.testPositivityRatioDetails.source",
    "metrics.caseDensity",
    "metrics.contactTracerCapacityRatio",
    "metrics.infectionRate",
    "metrics.infectionRateCI90",
    "metrics.icuHeadroomRatio",
    "metrics.icuHeadroomDetails.currentIcuCovid",
    "metrics.icuHeadroomDetails.currentIcuCovidMethod",
    "metrics.icuHeadroomDetails.currentIcuNonCovid",
    "metrics.icuHeadroomDetails.currentIcuNonCovidMethod",
    "metrics.icuCapacityRatio",
    "riskLevels.overall",
    "riskLevels.testPositivityRatio",
    "riskLevels.caseDensity",
    "riskLevels.contactTracerCapacityRatio",
    "riskLevels.infectionRate",
    "riskLevels.icuHeadroomRatio",
    "riskLevels.icuCapacityRatio",
    "actuals.cases",
    "actuals.deaths",
    "actuals.positiveTests",
    "actuals.negativeTests",
    "actuals.contactTracers",
    "actuals.hospitalBeds.capacity",
    "actuals.hospitalBeds.currentUsageTotal",
    "actuals.hospitalBeds.currentUsageCovid",
    "actuals.hospitalBeds.typicalUsageRate",
    "actuals.icuBeds.capacity",
    "actuals.icuBeds.currentUsageTotal",
    "actuals.icuBeds.currentUsageCovid",
    "actuals.icuBeds.typicalUsageRate",
    "actuals.newCases",
    "actuals.vaccinesDistributed",
    "actuals.vaccinationsInitiated",
    "actuals.vaccinationsCompleted",
    "lastUpdatedDate",
    "url",
    "metrics.vaccinationsInitiatedRatio",
    "metrics.vaccinationsCompletedRatio",
]

# Due to an inconsistency with how we previously were generating column names,
# state files had a different set of columns (that including headroom details). To prevent
# breaking changes, summary order with no headroom details is included here for County,
# metro, and place summary files.
SUMMARY_ORDER_NO_HEADROOM_DETAILS = [
    "fips",
    "country",
    "state",
    "county",
    "level",
    "lat",
    "locationId",
    "long",
    "population",
    "metrics.testPositivityRatio",
    "metrics.testPositivityRatioDetails.source",
    "metrics.caseDensity",
    "metrics.contactTracerCapacityRatio",
    "metrics.infectionRate",
    "metrics.infectionRateCI90",
    "metrics.icuHeadroomRatio",
    "metrics.icuHeadroomDetails",
    "metrics.icuCapacityRatio",
    "riskLevels.overall",
    "riskLevels.testPositivityRatio",
    "riskLevels.caseDensity",
    "riskLevels.contactTracerCapacityRatio",
    "riskLevels.infectionRate",
    "riskLevels.icuHeadroomRatio",
    "riskLevels.icuCapacityRatio",
    "actuals.cases",
    "actuals.deaths",
    "actuals.positiveTests",
    "actuals.negativeTests",
    "actuals.contactTracers",
    "actuals.hospitalBeds.capacity",
    "actuals.hospitalBeds.currentUsageTotal",
    "actuals.hospitalBeds.currentUsageCovid",
    "actuals.hospitalBeds.typicalUsageRate",
    "actuals.icuBeds.capacity",
    "actuals.icuBeds.currentUsageTotal",
    "actuals.icuBeds.currentUsageCovid",
    "actuals.icuBeds.typicalUsageRate",
    "actuals.newCases",
    "actuals.vaccinesDistributed",
    "actuals.vaccinationsInitiated",
    "actuals.vaccinationsCompleted",
    "lastUpdatedDate",
    "url",
    "metrics.vaccinationsInitiatedRatio",
    "metrics.vaccinationsCompletedRatio",
]


TIMESERIES_ORDER = [
    "date",
    "country",
    "state",
    "county",
    "fips",
    "lat",
    "long",
    "locationId",
    "actuals.cases",
    "actuals.deaths",
    "actuals.positiveTests",
    "actuals.negativeTests",
    "actuals.contactTracers",
    "actuals.hospitalBeds.capacity",
    "actuals.hospitalBeds.currentUsageTotal",
    "actuals.hospitalBeds.currentUsageCovid",
    "actuals.hospitalBeds.typicalUsageRate",
    "actuals.icuBeds.capacity",
    "actuals.icuBeds.currentUsageTotal",
    "actuals.icuBeds.currentUsageCovid",
    "actuals.icuBeds.typicalUsageRate",
    "actuals.newCases",
    "actuals.vaccinesDistributed",
    "actuals.vaccinationsInitiated",
    "actuals.vaccinationsCompleted",
    "metrics.testPositivityRatio",
    "metrics.testPositivityRatioDetails",
    "metrics.caseDensity",
    "metrics.contactTracerCapacityRatio",
    "metrics.infectionRate",
    "metrics.infectionRateCI90",
    "metrics.icuHeadroomRatio",
    "metrics.icuHeadroomDetails",
    "metrics.icuCapacityRatio",
    "riskLevels.overall",
    "metrics.vaccinationsInitiatedRatio",
    "metrics.vaccinationsCompletedRatio",
]
