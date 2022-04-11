from libs.datasets import data_source
from datapublic.common_fields import CommonFields
from libs.datasets.sources import can_scraper_helpers as ccd_helpers


class CDCCommunityLevelsDataset(data_source.CanScraperBase):
    SOURCE_TYPE = "CDCCommunityLevelsDataset"

    VARIABLES = [
        ccd_helpers.ScraperVariable(
            variable_name="cdc_community",
            measurement="current",
            unit="risk_level",
            provider="cdc_community_level",
            common_field=CommonFields.CDC_COMMUNITY_LEVEL,
        ),
        ccd_helpers.ScraperVariable(
            variable_name="hospital_beds_in_use_covid",
            measurement="rolling_average_7_day",
            unit="percentage",
            provider="cdc_community_level",
            common_field=CommonFields.BEDS_WITH_COVID_PATIENTS_RATIO_HSA,
        ),
        ccd_helpers.ScraperVariable(
            variable_name="hospital_admissions_covid",
            measurement="new_7_day",
            unit="people_per_100k",
            provider="cdc_community_level",
            common_field=CommonFields.WEEKLY_NEW_HOSPITAL_ADMISSIONS_COVID_PER_100K_HSA,
        ),
    ]
