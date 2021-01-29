import io
import pandas as pd
from libs.qa import data_availability


def test_build_availability_report():

    input_csv = [
        "fips,state,aggregate_level,population,field1,field2",
        "36061,NY,county,500,1,",
        "36062,NY,county,501,0,10",
        "36063,NY,county,501,0,15",
        "36,NY,state,10000,,10",
        "09,CT,state,10000,10,10",
    ]
    input_csv = "\n".join(input_csv)

    dataset = pd.read_csv(io.StringIO(input_csv))

    report = data_availability.build_data_availability_report(dataset)

    expected_csv = [
        "location_group,fips,population,field1,field2,num_locations",
        "state data,2,2,1,2,2",
        "NY,3,3,3,2,3",
    ]
    expected_csv = "\n".join(expected_csv)
    expected_df = pd.read_csv(io.StringIO(expected_csv))
    pd.testing.assert_frame_equal(expected_df, report.reset_index())
