from libs.datasets import data_source
from covidactnow.datapublic.common_fields import CommonFields
from libs.datasets.sources import can_scraper_helpers as ccd_helpers


def transform(dataset: ccd_helpers.CanScraperLoader):

    variables = [
        ccd_helpers.ScraperVariable(
            variable_name="total_vaccine_allocated",
            measurement="cumulative",
            unit="doses",
            provider="cdc",
            common_field=CommonFields.VACCINES_ALLOCATED,
        ),
        ccd_helpers.ScraperVariable(
            variable_name="total_vaccine_distributed",
            measurement="cumulative",
            unit="doses",
            provider="cdc",
            common_field=CommonFields.VACCINES_DISTRIBUTED,
        ),
        ccd_helpers.ScraperVariable(
            variable_name="total_vaccine_initiated",
            measurement="cumulative",
            unit="people",
            provider="cdc",
            common_field=CommonFields.VACCINATIONS_INITIATED,
        ),
        ccd_helpers.ScraperVariable(
            variable_name="total_vaccine_completed",
            measurement="cumulative",
            unit="people",
            provider="cdc",
            common_field=CommonFields.VACCINATIONS_COMPLETED,
        ),
    ]

    results = dataset.query_multiple_variables(variables)
    return results


class CDCVaccinesDataset(data_source.CanScraperBase):
    SOURCE_NAME = "CDCVaccine"

    TRANSFORM_METHOD = transform

    EXPECTED_FIELDS = [
        CommonFields.VACCINES_ALLOCATED,
        CommonFields.VACCINES_DISTRIBUTED,
        CommonFields.VACCINATIONS_INITIATED,
        CommonFields.VACCINATIONS_COMPLETED,
    ]
