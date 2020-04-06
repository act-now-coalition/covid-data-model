import logging
from collections import defaultdict
from libs import enums
import pandas as pd
from libs.datasets.beds import BedsDataset
from libs.datasets import dataset_utils
from libs.datasets import data_source

_logger = logging.getLogger(__name__)


def match_county_to_fips(data, fips_data, county_key="county", state_key="state"):
    county_combos = set(data.set_index([state_key, county_key]).index.to_list())
    fips_combos = set(
        fips_data.set_index([state_key, county_key, "fips"]).index.to_list()
    )
    fips_by_state_county = {
        (
            state,
            county.lower()
            .replace("'", "")
            .replace("-", " ")
            .replace("ó", "o")
            .replace("í", "i")
            .replace("é", "e")
            .replace("á", "a")
            .replace(".", "")
            .replace("ñ", ""),
        ): fips
        for state, county, fips in fips_combos
    }
    counties_by_state = defaultdict(list)

    for state, county, _ in fips_combos:
        counties_by_state[state].append(
            county.lower()
            .replace("'", "")
            .replace("-", " ")
            .replace("ó", "o")
            .replace("é", "e")
            .replace("í", "i")
            .replace("á", "a")
            .replace(".", "")
            .replace("ñ", "")
        )

    matched = {}
    suffixes = [
        "",
        " county",
        " parish",
        " municipio",
        " city and borough",
        " borough",
        " census area",
        " municipality",
        " city",
    ]
    replacements = {
        "de witt": "dewitt",
        "dewitt": "de witt",
        "la salle": "laselle",
        "dona ana": "doa ana",
        "la porte": "laporte",
        "de kalb": "dekalb",
    }
    for state, county in county_combos:
        key = state, county
        county = county.lower()
        county = replacements.get(county, county)
        if state in ["LA", "IL"] and county == "laselle":
            county = "lasalle"
        match = False
        for suffix in suffixes:
            with_suffix = county + suffix
            fips = fips_by_state_county.get((state, with_suffix))
            if fips:
                matched[key] = fips
                counties_by_state[state].remove(with_suffix)
                match = True
                break
            with_suffix = with_suffix.replace("saint ", "st ").replace(
                "sainte ", "ste "
            )
            fips = fips_by_state_county.get((state, with_suffix))
            if fips:
                matched[key] = fips
                counties_by_state[state].remove(with_suffix)
                match = True
                break

        if not match and state == 'VI':
            matched[key] = enums.UNKNOWN_FIPS
        elif not match:
            _logger.warning(f"Could not match {key}")
            if not counties_by_state[state]:
                continue

    records = [
        {"state": state, "county": county, "fips": fips}
        for (state, county), fips in matched.items()
    ]
    matches = pd.DataFrame(records)
    left = data.set_index(["state", "county"])
    right = matches.set_index(["state", "county"])
    return left.join(right).reset_index()


class DHBeds(data_source.DataSource):
    DATA_PATH = "data/beds-dh/hospital_beds_by_county.csv"
    SOURCE_NAME = "DH"

    class Fields(object):
        STATE = "state"
        COUNTY = "county"
        STAFFED_BEDS = "staffed_beds"
        LICENSED_BEDS = "licensed_beds"
        ICU_BEDS = "icu_beds"

        # Added in standardize data.
        AGGREGATE_LEVEL = "aggregate_level"
        FIPS = "fips"
        COUNTRY = "country"

    BEDS_FIELD_MAP = {
        BedsDataset.Fields.COUNTRY: Fields.COUNTRY,
        BedsDataset.Fields.STATE: Fields.STATE,
        BedsDataset.Fields.FIPS: Fields.FIPS,
        BedsDataset.Fields.STAFFED_BEDS: Fields.STAFFED_BEDS,
        BedsDataset.Fields.LICENSED_BEDS: Fields.LICENSED_BEDS,
        BedsDataset.Fields.ICU_BEDS: Fields.ICU_BEDS,
        BedsDataset.Fields.AGGREGATE_LEVEL: Fields.AGGREGATE_LEVEL,
    }

    def __init__(self, path):
        data = pd.read_csv(path)
        super().__init__(self.standardize_data(data))

    @classmethod
    def standardize_data(cls, data: pd.DataFrame) -> pd.DataFrame:
        # All DH data is aggregated at the county level
        data[cls.Fields.AGGREGATE_LEVEL] = "county"

        data[cls.Fields.COUNTRY] = "USA"

        # Backfilling FIPS data based on county names.
        fips_data = dataset_utils.build_fips_data_frame()
        data = match_county_to_fips(data, fips_data)

        # The virgin islands do not currently have associated fips codes.
        # if VI is supported in the future, this should be removed.
        is_virgin_islands = data[cls.Fields.STATE] == 'VI'
        return data[~is_virgin_islands]

    @classmethod
    def local(cls) -> "DHBeds":
        data_root = dataset_utils.LOCAL_PUBLIC_DATA_PATH
        return cls(data_root / cls.DATA_PATH)
