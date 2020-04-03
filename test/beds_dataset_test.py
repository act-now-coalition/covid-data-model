import pandas as pd
from libs.datasets import beds


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
        "model_input_beds": 21480,
        "county": None,
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
        "model_input_beds": 4474,
        "county": "Middlesex County",
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
        generate_county_bed_row(fips="25016", county=county, model_input_beds=3000),
        generate_county_bed_row(fips=fips, model_input_beds=1000),
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
        generate_county_bed_row(model_input_beds=2000),
        generate_state_bed_row(model_input_beds=1000),
        generate_state_bed_row(state="CA", model_input_beds=2000),
    ]
    beds_dataset = build_beds_dataset(rows)

    state_beds = beds_dataset.get_state_level("MA")
    assert state_beds == 1000

    assert not beds_dataset.get_state_level("DC")
