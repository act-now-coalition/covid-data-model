from typing import Tuple
import io
import shapefile
from urllib.parse import urlparse

from libs.CovidDatasets import get_public_data_base_url
from libs.constants import NULL_VALUE
from libs.datasets.results_schema import NON_INTEGER_FIELDS, SHAPEFILE_FIELDS

def _file_uri_to_path(uri: str) -> str:
    return urlparse(uri).path


def join_and_output_shapefile(
    df, shp_reader, pivot_shp_field, pivot_df_column, shp_writer
):
    shp_writer.fields = shp_reader.fields  # Preserve fields that come from the census

    for field_name in SHAPEFILE_FIELDS:
        if field_name in NON_INTEGER_FIELDS:
            shp_writer.field(field_name, "C", size=32)
        else:
            shp_writer.field(field_name, "N", size=14)

    shapeRecords = shp_reader.shapeRecords()
    # if you are using a local copy of the data, LFS truncates the records
    assert len(shapeRecords) >= 50

    # Just adding some understanding of errors
    failed_dictionary = {}

    for shapeRecord in shapeRecords:
        try:
            # Gets the row of the dataframe that matches the FIPS codes for a state/county
            row = df[df[pivot_df_column] == shapeRecord.record[pivot_shp_field]].iloc[0]
        except Exception as e:
            state_fips = shapeRecord.record[pivot_shp_field][
                :2
            ]  # state fips is the first two chars of the state/county fips
            failed_dictionary.setdefault(state_fips, []).append(
                shapeRecord.record[pivot_shp_field]
            )
            continue

        # create record data for all the fields create a shape record
        new_record = shapeRecord.record.as_dict()
        for field_name in SHAPEFILE_FIELDS:
            new_record[field_name] = (
                None if row[field_name] == NULL_VALUE else row[field_name]
            )
        shp_writer.shape(shapeRecord.shape)
        shp_writer.record(**new_record)

    # Different errors in each state, note this includes the territories as well
    print([(state, len(failed_dictionary[state])) for state in failed_dictionary])
    shp_writer.close()


def get_usa_state_shapefile(use_state_df) -> Tuple[io.BytesIO, io.BytesIO, io.BytesIO]:
    shp = io.BytesIO()
    shx = io.BytesIO()
    dbf = io.BytesIO()

    shp_writer = shapefile.Writer(shp=shp, shx=shx, dbf=dbf)
    public_data_url = get_public_data_base_url()
    public_data_path = _file_uri_to_path(public_data_url)
    join_and_output_shapefile(
        use_state_df,
        shapefile.Reader(
            f"{public_data_path}/data/shapefiles-uscensus/tl_2019_us_state"
        ),
        "STATEFP",
        "State/County FIPS Code",
        shp_writer,
    )
    return shp, shx, dbf


def get_usa_county_shapefile(county_df) -> Tuple[io.BytesIO, io.BytesIO, io.BytesIO]:
    shp = io.BytesIO()
    shx = io.BytesIO()
    dbf = io.BytesIO()

    shp_writer = shapefile.Writer(shp=shp, shx=shx, dbf=dbf)
    public_data_url = get_public_data_base_url()
    public_data_path = _file_uri_to_path(public_data_url)

    join_and_output_shapefile(
        county_df,
        shapefile.Reader(
            f"{public_data_path}/data/shapefiles-uscensus/tl_2019_us_county"
        ),
        "GEOID",
        "State/County FIPS Code",
        shp_writer,
    )
    return shp, shx, dbf
