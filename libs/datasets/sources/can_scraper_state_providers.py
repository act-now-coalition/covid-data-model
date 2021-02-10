from libs.datasets import data_source
from covidactnow.datapublic.common_fields import CommonFields
from libs.datasets.sources import can_scraper_helpers as ccd_helpers


def transform(dataset: ccd_helpers.CanScraperLoader):
    variables = [
        ccd_helpers.ScraperVariable(variable_name="pcr_tests_negative", provider="state"),
        ccd_helpers.ScraperVariable(variable_name="unspecified_tests_total", provider="state"),
        ccd_helpers.ScraperVariable(variable_name="unspecified_tests_positive", provider="state"),
        ccd_helpers.ScraperVariable(variable_name="icu_beds_available", provider="state"),
        ccd_helpers.ScraperVariable(variable_name="antibody_tests_total", provider="state"),
        ccd_helpers.ScraperVariable(variable_name="antigen_tests_positive", provider="state"),
        ccd_helpers.ScraperVariable(variable_name="antigen_tests_negative", provider="state"),
        ccd_helpers.ScraperVariable(
            variable_name="total_vaccine_doses_administered", provider="state"
        ),
        ccd_helpers.ScraperVariable(variable_name="hospital_beds_in_use", provider="state"),
        ccd_helpers.ScraperVariable(variable_name="ventilators_in_use", provider="state"),
        ccd_helpers.ScraperVariable(variable_name="ventilators_available", provider="state"),
        ccd_helpers.ScraperVariable(variable_name="ventilators_capacity", provider="state"),
        ccd_helpers.ScraperVariable(variable_name="pediatric_icu_beds_in_use", provider="state"),
        ccd_helpers.ScraperVariable(variable_name="adult_icu_beds_available", provider="state"),
        ccd_helpers.ScraperVariable(variable_name="pediatric_icu_beds_capacity", provider="state"),
        ccd_helpers.ScraperVariable(variable_name="unspecified_tests_negative", provider="state"),
        ccd_helpers.ScraperVariable(variable_name="antigen_tests_total", provider="state"),
        ccd_helpers.ScraperVariable(variable_name="adult_icu_beds_in_use", provider="state"),
        ccd_helpers.ScraperVariable(variable_name="hospital_beds_available", provider="state"),
        ccd_helpers.ScraperVariable(variable_name="pediatric_icu_beds_available", provider="state"),
        ccd_helpers.ScraperVariable(variable_name="adult_icu_beds_capacity", provider="state"),
        ccd_helpers.ScraperVariable(variable_name="icu_beds_in_use", provider="state"),
        ccd_helpers.ScraperVariable(
            variable_name="cases",
            measurement="cumulative",
            unit="people",
            provider="state",
            common_field=CommonFields.CASES,
        ),
        ccd_helpers.ScraperVariable(
            variable_name="deaths",
            measurement="cumulative",
            unit="people",
            provider="state",
            common_field=CommonFields.DEATHS,
        ),
        ccd_helpers.ScraperVariable(
            variable_name="hospital_beds_in_use_covid",
            measurement="current",
            unit="beds",
            provider="state",
            common_field=CommonFields.CURRENT_HOSPITALIZED,
        ),
        ccd_helpers.ScraperVariable(
            variable_name="hospital_beds_capacity",
            measurement="current",
            unit="beds",
            provider="state",
            common_field=CommonFields.STAFFED_BEDS,
        ),
        ccd_helpers.ScraperVariable(
            variable_name="icu_beds_capacity",
            measurement="current",
            unit="beds",
            provider="state",
            common_field=CommonFields.ICU_BEDS,
        ),
        ccd_helpers.ScraperVariable(
            variable_name="icu_beds_in_use_covid",
            measurement="current",
            unit="beds",
            provider="state",
            common_field=CommonFields.CURRENT_ICU,
        ),
        ccd_helpers.ScraperVariable(
            variable_name="pcr_tests_total",
            measurement="cumulative",
            unit="specimens",  # Ignores less common unit=test_encounters and unit=unique_people
            provider="state",
            common_field=CommonFields.TOTAL_TESTS_VIRAL,
        ),
        ccd_helpers.ScraperVariable(
            variable_name="pcr_tests_positive",
            measurement="cumulative",
            unit="specimens",  # Ignores test_encounters and unique_people
            provider="state",
            common_field=CommonFields.POSITIVE_TESTS_VIRAL,
        ),
        ccd_helpers.ScraperVariable(
            variable_name="total_vaccine_allocated",
            measurement="cumulative",
            unit="doses",
            provider="state",
            common_field=CommonFields.VACCINES_ALLOCATED,
        ),
        ccd_helpers.ScraperVariable(
            variable_name="total_vaccine_distributed",
            measurement="cumulative",
            unit="doses",
            provider="state",
            common_field=CommonFields.VACCINES_DISTRIBUTED,
        ),
        ccd_helpers.ScraperVariable(
            variable_name="total_vaccine_initiated",
            measurement="cumulative",
            unit="people",
            provider="state",
            common_field=CommonFields.VACCINATIONS_INITIATED,
        ),
        ccd_helpers.ScraperVariable(
            variable_name="total_vaccine_initiated",
            measurement="current",
            unit="percentage",
            provider="state",
            common_field=CommonFields.VACCINATIONS_INITIATED_PCT,
        ),
        ccd_helpers.ScraperVariable(
            variable_name="total_vaccine_completed",
            measurement="cumulative",
            unit="people",
            provider="state",
            common_field=CommonFields.VACCINATIONS_COMPLETED,
        ),
        ccd_helpers.ScraperVariable(
            variable_name="total_vaccine_completed",
            measurement="current",
            unit="percentage",
            provider="state",
            common_field=CommonFields.VACCINATIONS_COMPLETED_PCT,
        ),
        ccd_helpers.ScraperVariable(
            variable_name="total_vaccine_doses_administered",
            measurement="cumulative",
            unit="doses",
            provider="state",
            common_field=CommonFields.VACCINES_ADMINISTERED,
        ),
    ]

    results = dataset.query_multiple_variables(variables, log_provider_coverage_warnings=True)
    return results


class CANScraperStateProviders(data_source.CanScraperBase):
    SOURCE_NAME = "CANScrapersStateProviders"

    TRANSFORM_METHOD = transform

    EXPECTED_FIELDS = [
        CommonFields.STAFFED_BEDS,
        CommonFields.CASES,
        CommonFields.DEATHS,
        CommonFields.VACCINES_ALLOCATED,
        CommonFields.VACCINES_ADMINISTERED,
        CommonFields.VACCINES_DISTRIBUTED,
        CommonFields.VACCINATIONS_INITIATED,
        CommonFields.VACCINATIONS_COMPLETED,
        CommonFields.TOTAL_TESTS_VIRAL,
        CommonFields.ICU_BEDS,
        CommonFields.CURRENT_HOSPITALIZED,
        CommonFields.POSITIVE_TESTS_VIRAL,
        CommonFields.CURRENT_ICU,
        CommonFields.VACCINATIONS_INITIATED_PCT,
        CommonFields.VACCINATIONS_COMPLETED_PCT,
    ]
