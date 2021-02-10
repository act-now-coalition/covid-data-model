from covidactnow.datapublic.common_fields import CommonFields

from libs.datasets.sources import can_scraper_helpers as ccd_helpers

from libs.datasets import data_source


def transform_cases_and_deaths(dataset: ccd_helpers.CovidCountyDataset):
    variables = [
        ccd_helpers.ScraperVariable(
            variable_name="cases",
            measurement="cumulative",
            unit="people",
            provider="usafacts",
            common_field=CommonFields.CASES,
        ),
        ccd_helpers.ScraperVariable(
            variable_name="deaths",
            measurement="cumulative",
            unit="people",
            provider="usafacts",
            common_field=CommonFields.DEATHS,
        ),
    ]
    results = dataset.query_multiple_variables(variables, log_provider_coverage_warnings=True)
    return results


class CANScraperUSAFactsProvider(data_source.CanScraperBase):
    SOURCE_NAME = "USAFacts"

    TRANSFORM_METHOD = transform_cases_and_deaths

    EXPECTED_FIELDS = [
        CommonFields.CASES,
        CommonFields.DEATHS,
    ]
