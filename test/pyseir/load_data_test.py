from datetime import datetime

from libs.pipeline import Region
from pyseir import load_data
from pyseir.load_data import HospitalizationCategory
from pyseir.load_data import HospitalizationDataType


def test_load_hospitalization_data():
    t0 = datetime(year=2020, month=1, day=1)
    region = Region.from_fips("33")
    hospitalization_dataset = load_data.get_hospitalization_data().get_one_region(region)

    _, _, hosp_type = load_data.calculate_hospitalization_data(
        hospitalization_dataset, t0, category=HospitalizationCategory.ICU
    )
    # Double check that data loads and it went throughh the cumulative hosps
    assert hosp_type is HospitalizationDataType.CUMULATIVE_HOSPITALIZATIONS
