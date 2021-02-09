from scripts import update_can_scraper_state_providers

from libs.datasets import data_source


class CANScraperStateProviders(data_source.CanScraperBase):
    SOURCE_NAME = "CANScrapersStateProviders"

    TRANSFORM_METHOD = update_can_scraper_state_providers.transform
