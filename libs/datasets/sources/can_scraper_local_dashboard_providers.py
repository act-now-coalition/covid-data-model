from libs.datasets import data_source
from datapublic.common_fields import CommonFields
from libs.datasets.sources import can_scraper_helpers as ccd_helpers


# TODO(michael): Consider centralizing this mapping of
# variable+measurement+unit => common_field so that it can be reused across
# other datasets more easily.
def make_scraper_variables(provider: str):
    """Helper to generate all variables that could be captured from a state / county dashboard."""
    return [
        ccd_helpers.ScraperVariable(variable_name="pcr_tests_negative", provider=provider),
        ccd_helpers.ScraperVariable(variable_name="unspecified_tests_total", provider=provider),
        ccd_helpers.ScraperVariable(variable_name="unspecified_tests_positive", provider=provider),
        ccd_helpers.ScraperVariable(variable_name="icu_beds_available", provider=provider),
        ccd_helpers.ScraperVariable(variable_name="antibody_tests_total", provider=provider),
        ccd_helpers.ScraperVariable(variable_name="antigen_tests_positive", provider=provider),
        ccd_helpers.ScraperVariable(variable_name="antigen_tests_negative", provider=provider),
        ccd_helpers.ScraperVariable(
            variable_name="total_vaccine_doses_administered", provider=provider
        ),
        ccd_helpers.ScraperVariable(variable_name="hospital_beds_in_use", provider=provider),
        ccd_helpers.ScraperVariable(variable_name="ventilators_in_use", provider=provider),
        ccd_helpers.ScraperVariable(variable_name="ventilators_available", provider=provider),
        ccd_helpers.ScraperVariable(variable_name="ventilators_capacity", provider=provider),
        ccd_helpers.ScraperVariable(variable_name="pediatric_icu_beds_in_use", provider=provider),
        ccd_helpers.ScraperVariable(variable_name="adult_icu_beds_available", provider=provider),
        ccd_helpers.ScraperVariable(variable_name="pediatric_icu_beds_capacity", provider=provider),
        ccd_helpers.ScraperVariable(variable_name="unspecified_tests_negative", provider=provider),
        ccd_helpers.ScraperVariable(variable_name="antigen_tests_total", provider=provider),
        ccd_helpers.ScraperVariable(variable_name="adult_icu_beds_in_use", provider=provider),
        ccd_helpers.ScraperVariable(variable_name="hospital_beds_available", provider=provider),
        ccd_helpers.ScraperVariable(
            variable_name="pediatric_icu_beds_available", provider=provider
        ),
        ccd_helpers.ScraperVariable(variable_name="adult_icu_beds_capacity", provider=provider),
        ccd_helpers.ScraperVariable(variable_name="icu_beds_in_use", provider=provider),
        ccd_helpers.ScraperVariable(
            variable_name="cases",
            measurement="cumulative",
            unit="people",
            provider=provider,
            common_field=CommonFields.CASES,
        ),
        ccd_helpers.ScraperVariable(
            variable_name="deaths",
            measurement="cumulative",
            unit="people",
            provider=provider,
            common_field=CommonFields.DEATHS,
        ),
        ccd_helpers.ScraperVariable(
            variable_name="hospital_beds_in_use_covid",
            measurement="current",
            unit="beds",
            provider=provider,
            common_field=CommonFields.CURRENT_HOSPITALIZED,
        ),
        ccd_helpers.ScraperVariable(
            variable_name="hospital_beds_capacity",
            measurement="current",
            unit="beds",
            provider=provider,
            common_field=CommonFields.STAFFED_BEDS,
        ),
        ccd_helpers.ScraperVariable(
            variable_name="icu_beds_capacity",
            measurement="current",
            unit="beds",
            provider=provider,
            common_field=CommonFields.ICU_BEDS,
        ),
        ccd_helpers.ScraperVariable(
            variable_name="icu_beds_in_use_covid",
            measurement="current",
            unit="beds",
            provider=provider,
            common_field=CommonFields.CURRENT_ICU,
        ),
        ccd_helpers.ScraperVariable(
            variable_name="pcr_tests_total",
            measurement="cumulative",
            unit="specimens",  # Ignores less common unit=test_encounters and unit=unique_people
            provider=provider,
            common_field=CommonFields.TOTAL_TESTS_VIRAL,
        ),
        ccd_helpers.ScraperVariable(
            variable_name="pcr_tests_positive",
            measurement="cumulative",
            unit="specimens",  # Ignores test_encounters and unique_people
            provider=provider,
            common_field=CommonFields.POSITIVE_TESTS_VIRAL,
        ),
        ccd_helpers.ScraperVariable(
            variable_name="total_vaccine_allocated",
            measurement="cumulative",
            unit="doses",
            provider=provider,
            common_field=CommonFields.VACCINES_ALLOCATED,
        ),
        ccd_helpers.ScraperVariable(
            variable_name="total_vaccine_distributed",
            measurement="cumulative",
            unit="doses",
            provider=provider,
            common_field=CommonFields.VACCINES_DISTRIBUTED,
        ),
        ccd_helpers.ScraperVariable(
            variable_name="total_vaccine_initiated",
            measurement="cumulative",
            unit="people",
            provider=provider,
            common_field=CommonFields.VACCINATIONS_INITIATED,
        ),
        ccd_helpers.ScraperVariable(
            variable_name="total_vaccine_initiated",
            measurement="current",
            unit="percentage",
            provider=provider,
            common_field=CommonFields.VACCINATIONS_INITIATED_PCT,
        ),
        ccd_helpers.ScraperVariable(
            variable_name="total_vaccine_completed",
            measurement="cumulative",
            unit="people",
            provider=provider,
            common_field=CommonFields.VACCINATIONS_COMPLETED,
        ),
        ccd_helpers.ScraperVariable(
            variable_name="total_vaccine_additional_dose",
            measurement="cumulative",
            unit="people",
            provider=provider,
            common_field=CommonFields.VACCINATIONS_ADDITIONAL_DOSE,
        ),
        ccd_helpers.ScraperVariable(
            variable_name="total_vaccine_completed",
            measurement="current",
            unit="percentage",
            provider=provider,
            common_field=CommonFields.VACCINATIONS_COMPLETED_PCT,
        ),
        ccd_helpers.ScraperVariable(
            variable_name="total_vaccine_doses_administered",
            measurement="cumulative",
            unit="doses",
            provider=provider,
            common_field=CommonFields.VACCINES_ADMINISTERED,
        ),
    ]


class CANScraperStateProviders(data_source.CanScraperBase):
    SOURCE_TYPE = "CANScrapersStateProviders"

    VARIABLES = make_scraper_variables("state")


class CANScraperCountyProviders(data_source.CanScraperBase):
    SOURCE_TYPE = "CANScrapersCountyProviders"

    VARIABLES = make_scraper_variables("county")
