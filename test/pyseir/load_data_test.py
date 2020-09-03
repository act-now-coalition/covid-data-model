from datetime import datetime
from pyseir import load_data
from pyseir.load_data import HospitalizationCategory
from pyseir.load_data import HospitalizationDataType


def test_load_hospitalization_data():
    t0 = datetime(year=2020, month=1, day=1)
    fips = "33"
    hospitalization_df = load_data.get_hospitalization_data().get_data(fips=fips)

    _, _, hosp_type = load_data.calculate_hospitalization_data(
        hospitalization_df, t0, category=HospitalizationCategory.ICU
    )
    # Double check that data loads and it went throughh the cumulative hosps
    assert hosp_type is HospitalizationDataType.CUMULATIVE_HOSPITALIZATIONS
