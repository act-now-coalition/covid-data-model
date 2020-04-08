import pytest
import pandas as pd
from libs.datasets import beds
from libs.datasets import dataset_utils


def generate_state_bed_row(**updates):
    record = {
        "fips": None,
        "state": "MA",
        "staffed_beds": 21374,
        "licensed_beds": 18280,
        "icu_beds": 1223,
        "aggregate_level": "state",
        "source": "DH",
        "generated": True,
        "max_bed_count": 21480,
        "county": None,
        "country": "USA",
    }
    record.update(**updates)
    return record


def generate_county_bed_row(**updates):
    record = {
        "fips": "25017",
        "state": "MA",
        "staffed_beds": 4474,
        "licensed_beds": 3756,
        "icu_beds": 206,
        "aggregate_level": "county",
        "source": "DH",
        "generated": False,
        "max_bed_count": 4474,
        "county": "Middlesex County",
        "country": "USA",
    }
    record.update(**updates)
    return record


def build_beds_dataset(rows):
    data = pd.DataFrame(rows)
    return beds.BedsDataset(data)


def test_get_county_level_beds():
    fips = "25015"
    county = "County in Mass"
    rows = [
        generate_county_bed_row(fips="25016", county=county, max_bed_count=3000),
        generate_county_bed_row(fips=fips, max_bed_count=1000),
        generate_state_bed_row(),
    ]
    beds_dataset = build_beds_dataset(rows)

    county_beds = beds_dataset.get_county_level("MA", county=county)
    assert county_beds == 3000

    county_beds = beds_dataset.get_county_level("MA", fips=fips)
    assert county_beds == 1000

    county_beds = beds_dataset.get_county_level("MA", fips="random")
    assert not county_beds


def test_get_state_level_beds():
    rows = [
        generate_county_bed_row(max_bed_count=2000),
        generate_state_bed_row(max_bed_count=1000),
        generate_state_bed_row(state="CA", max_bed_count=2000),
    ]
    beds_dataset = build_beds_dataset(rows)

    state_beds = beds_dataset.get_state_level("MA")
    assert state_beds == 1000

    assert not beds_dataset.get_state_level("DC")


@pytest.mark.parametrize("is_county", [True, False])
def test_duplicate_index_fails(is_county):
    if is_county:
        rows = [
            generate_county_bed_row(),
            generate_county_bed_row(),
        ]
    else:
        rows = [
            generate_state_bed_row(),
            generate_state_bed_row(),
        ]

    with pytest.raises(dataset_utils.DuplicateValuesForIndex):
        build_beds_dataset(rows)
