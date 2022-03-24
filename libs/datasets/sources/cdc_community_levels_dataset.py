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
        # Note: we don't import the rest of the fields
        # (covid_inpatient_bed_utilization, covid_hospital_admissions_per_100k,
        # covid_cases_per_100k) since we don't currently use the CDC data.
    ]
