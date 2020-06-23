import pathlib
import json
import pandas as pd
from libs.datasets import dataset_utils
from libs.datasets import data_source
from libs import build_params


def _category_list_to_dict(categories):
    data = {}
    for category in categories:
        data[category["category"]] = category.get("percent", 0) / 100.0

    return data


def parse_state_mobility_json(data) -> pd.DataFrame:
    parsed_data = []
    for record in data:
        country_code = record["country_code"]
        if not country_code.startswith("US_"):
            continue

        country, *state_long = country_code.split("_")
        state_long = " ".join(state_long)
        if country == "US":
            country = "USA"
        state = build_params.US_STATE_ABBREV[state_long]
        categories = _category_list_to_dict(record["national_data"])

        state_data = {"country": country, "state": state, "aggregate_level": "state"}
        state_data.update(categories)
        parsed_data.append(state_data)

        for county_record in record["county_data"]:
            county = county_record["county_name"]
            county_category = _category_list_to_dict(county_record["data"])
            county_record = {
                "country": country,
                "state": state,
                "county": county,
                "aggregate_level": "county",
            }
            county_record.update(county_category)
            parsed_data.append(county_record)

    return pd.DataFrame(parsed_data)


class GoogleMobilityReport(data_source.DataSource):
    DATA_PATH = "data/google-mobility/mobility_scrape.json"

    class Fields(object):
        STATE = "state"
        COUNTY = "county"
        COUNTRY = "country"

        RESIDENTIAL = "Residential"
        WORKPLACES = "Workplaces"
        TRANSIT_STATIONS = "Transit stations"
        PARKS = "Parks"
        GROCERY = "Grocery & pharmacy"
        RETAIL_RECREATION = "Retail & recreation"
        AGGREGATE_LEVEL = "aggregate_level"

        FIPS = "fips"

    def __init__(self, path: pathlib.Path):
        data = json.load(path.open())
        data = parse_state_mobility_json(data)
        self.data = self.standardize_data(data)

    @classmethod
    def standardize_data(cls, data: pd.DataFrame) -> pd.DataFrame:
        def replace_missing_l(county):
            # for some reason doubly repeated letters weren't parsed correctly...
            # this fixes that.
            if pd.isna(county):
                return county

            missing_l_combos = ["l e", "l a", "l i", "l o", "l s", "l m"]
            for missing_l in missing_l_combos:
                if missing_l in county:
                    replace_with = missing_l.replace(" ", "l")
                    return county.replace(missing_l, replace_with)

            if "l  " in county:
                return county.replace("l  ", "ll ")
            if "i  " in county:
                return county.replace("i  ", "ii ")

            return county

        data[cls.Fields.COUNTY] = data[cls.Fields.COUNTY].apply(replace_missing_l)
        fips_data = dataset_utils.build_fips_data_frame()
        data = dataset_utils.add_fips_using_county(data, fips_data)
        return data

    @classmethod
    def local(cls) -> "DHBeds":
        data_root = dataset_utils.LOCAL_PUBLIC_DATA_PATH
        return cls(data_root / cls.DATA_PATH)
