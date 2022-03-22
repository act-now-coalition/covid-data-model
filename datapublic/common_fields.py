"""
Data schema shared between code in covid-data-public and covid-data-model repos.
"""
import enum
from typing import Optional


class GetByValueMixin:
    """Mixin making it easy to get an Enum object or None if not found.

    Unlike `YourEnumClass(value)`, the `get` method does not raise `ValueError` when `value`
    is not in the enum.
    """

    @classmethod
    def get(cls, value):
        return cls._value2member_map_.get(value, None)


class ValueAsStrMixin:
    def __str__(self):
        # Make sure str(CommonFields.CASES) returns a str, not a FieldName. DataFrame.itertuples
        # passes a list of fields to collections.namedtuple which calls map(str, fields) and then
        # checks that the result types are str. Until Python 3.8.5 returning self.value here
        # worked, but starting with 3.8.6 str would produce a FieldName causing the type
        # check in namedtuple to fail.
        return str(self.value)


class FieldName(str):
    """Common base-class for enums of fields, CSV column names etc"""

    __reduce_ex__ = str.__reduce_ex__  # Work-around for https://bugs.python.org/issue44342


class DemographicBucket(str):
    """Represents a demographic bucket name such as "all" or "age:10-19" or "age:60-69;sex:female".
    These are not enumerated in this repo but there is a list of them at
    https://github.com/covid-projections/can-scrapers/blob/main/can_tools/bootstrap_data/covid_demographics.csv"""

    ALL: "DemographicBucket"


DemographicBucket.ALL = DemographicBucket("all")


@enum.unique
class FieldGroup(ValueAsStrMixin, str, enum.Enum):
    TESTS = "tests"
    VACCINES = "vaccines"
    HEALTHCARE_CAPACITY = "hospitals"
    CASES_DEATHS = "cases_deaths"


@enum.unique
class CommonFields(GetByValueMixin, ValueAsStrMixin, FieldName, enum.Enum):
    """Common field names shared across different sources of data"""

    def __new__(cls, field_name: str, field_group: Optional[FieldGroup]):
        """Make an enum that is both a str and has a field_group attribute."""
        # TODO(tom): When we are using Python >= 3.8 incorporate backports.strenum, per
        #  https://discuss.python.org/t/built-in-strenum/4192
        # Could be replaced with use of https://pypi.org/project/typeguard/
        assert isinstance(field_name, str)
        assert field_group is None or isinstance(field_group, FieldGroup)
        # This solution is hacked from https://stackoverflow.com/a/49286691
        o = super().__new__(cls, field_name)
        # _value_ is a special name of enum. Set it here so enum code doesn't attempt to call
        # str(field_name, field_group).
        o._value_ = field_name
        o.field_group = field_group
        return o

    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))

    FIPS = "fips", None

    DATE = "date", None

    # In the style of CovidAtlas/Project Li `locationID`. See
    # https://github.com/covidatlas/li/blob/master/docs/reports-v1.md#general-notes
    LOCATION_ID = "location_id", None

    # 2 letter state abbreviation, i.e. MA
    STATE = "state", None

    COUNTRY = "country", None

    COUNTY = "county", None

    HSA = "hsa", None

    AGGREGATE_LEVEL = "aggregate_level", None

    # Full state name, i.e. Massachusetts
    STATE_FULL_NAME = "state_full_name", None

    CASES = "cases", FieldGroup.CASES_DEATHS
    DEATHS = "deaths", FieldGroup.CASES_DEATHS
    RECOVERED = "recovered", FieldGroup.CASES_DEATHS

    # Incidence Values
    NEW_CASES = "new_cases", FieldGroup.CASES_DEATHS
    NEW_DEATHS = "new_deaths", FieldGroup.CASES_DEATHS
    WEEKLY_NEW_CASES = "weekly_new_cases", FieldGroup.CASES_DEATHS
    WEEKLY_NEW_DEATHS = "weekly_new_deaths", FieldGroup.CASES_DEATHS

    # Forecast Specific Columns
    MODEL_ABBR = "model_abbr", None  # The label of the model used for prediction
    FORECAST_DATE = "forecast_date", None  # The prediction made with data up to that date
    QUANTILE = "quantile", None  # Prediction Levels

    # Cumulative values
    CUMULATIVE_HOSPITALIZED = "cumulative_hospitalized", FieldGroup.HEALTHCARE_CAPACITY
    CUMULATIVE_ICU = "cumulative_icu", FieldGroup.HEALTHCARE_CAPACITY

    POSITIVE_TESTS = "positive_tests", FieldGroup.TESTS
    NEGATIVE_TESTS = "negative_tests", FieldGroup.TESTS
    TOTAL_TESTS = "total_tests", FieldGroup.TESTS

    POSITIVE_TESTS_VIRAL = "positive_tests_viral", FieldGroup.TESTS
    POSITIVE_CASES_VIRAL = "positive_cases_viral", FieldGroup.TESTS
    TOTAL_TESTS_VIRAL = "total_tests_viral", FieldGroup.TESTS
    TOTAL_TESTS_PEOPLE_VIRAL = "total_tests_people_viral", FieldGroup.TESTS
    TOTAL_TEST_ENCOUNTERS_VIRAL = "total_test_encounters_viral", FieldGroup.TESTS

    # Current values
    CURRENT_ICU = "current_icu", FieldGroup.HEALTHCARE_CAPACITY
    CURRENT_HOSPITALIZED = "current_hospitalized", FieldGroup.HEALTHCARE_CAPACITY
    CURRENT_VENTILATED = "current_ventilated", FieldGroup.HEALTHCARE_CAPACITY

    POPULATION = "population", None
    HSA_POPULATION = "hsa_population", None

    STAFFED_BEDS = "staffed_beds", FieldGroup.HEALTHCARE_CAPACITY
    LICENSED_BEDS = "licensed_beds", FieldGroup.HEALTHCARE_CAPACITY
    ICU_BEDS = "icu_beds", FieldGroup.HEALTHCARE_CAPACITY
    ALL_BED_TYPICAL_OCCUPANCY_RATE = "all_beds_occupancy_rate", FieldGroup.HEALTHCARE_CAPACITY
    ICU_TYPICAL_OCCUPANCY_RATE = "icu_occupancy_rate", FieldGroup.HEALTHCARE_CAPACITY
    MAX_BED_COUNT = "max_bed_count", FieldGroup.HEALTHCARE_CAPACITY
    VENTILATOR_CAPACITY = "ventilator_capacity", FieldGroup.HEALTHCARE_CAPACITY

    HOSPITAL_BEDS_IN_USE_ANY = "hospital_beds_in_use_any", FieldGroup.HEALTHCARE_CAPACITY
    CURRENT_HOSPITALIZED_TOTAL = "current_hospitalized_total", FieldGroup.HEALTHCARE_CAPACITY
    CURRENT_ICU_TOTAL = "current_icu_total", FieldGroup.HEALTHCARE_CAPACITY

    CONTACT_TRACERS_COUNT = "contact_tracers_count", FieldGroup.HEALTHCARE_CAPACITY
    LATITUDE = "latitude", None
    LONGITUDE = "longitude", None

    # Ratio of positive tests to total tests, from 0.0 to 1.0
    TEST_POSITIVITY = "test_positivity", FieldGroup.TESTS
    TEST_POSITIVITY_14D = "test_positivity_14d", FieldGroup.TESTS
    TEST_POSITIVITY_7D = "test_positivity_7d", FieldGroup.TESTS

    CAN_LOCATION_PAGE_URL = "can_location_page_url", None

    # vaccines_ prefixed variables are in doses of vaccines
    VACCINES_ALLOCATED = "vaccines_allocated", FieldGroup.VACCINES
    VACCINES_DISTRIBUTED = "vaccines_distributed", FieldGroup.VACCINES
    VACCINES_ADMINISTERED = "vaccines_administered", FieldGroup.VACCINES

    # vaccinations_ prefixed variables are people vaccinated.
    # _pct suffix is percent of the population which may be copied from an external source,
    # for example
    # https://github.com/covid-projections/can-scrapers/blob/main/can_tools/scrapers/official/TN/tn_vaccine.py
    # or derived from the count of people vaccinated.
    VACCINATIONS_INITIATED = "vaccinations_initiated", FieldGroup.VACCINES
    VACCINATIONS_INITIATED_PCT = "vaccinations_initiated_pct", FieldGroup.VACCINES
    VACCINATIONS_ADDITIONAL_DOSE = "vaccinations_additional_dose", FieldGroup.VACCINES
    VACCINATIONS_ADDITIONAL_DOSE_PCT = "vaccinations_additional_dose_pct", FieldGroup.VACCINES
    VACCINATIONS_COMPLETED = "vaccinations_completed", FieldGroup.VACCINES
    VACCINATIONS_COMPLETED_PCT = "vaccinations_completed_pct", FieldGroup.VACCINES


COMMON_FIELD_TO_GROUP = {f: f.field_group for f in CommonFields if f.field_group}

FIELD_GROUP_TO_LIST_FIELDS = {}
for field, field_group in COMMON_FIELD_TO_GROUP.items():
    FIELD_GROUP_TO_LIST_FIELDS.setdefault(field_group, []).append(field)


@enum.unique
class PdFields(GetByValueMixin, ValueAsStrMixin, FieldName, enum.Enum):
    """Field names that are used in Pandas but not directly related to COVID metrics"""

    # Identifies the metric or variable name in Panda DataFrames with only one value ('long' layout) or
    # timeseries ('date wide' layout) per row.
    VARIABLE = "variable"
    # Column containing the value in 'long' format DataFrames.
    VALUE = "value"

    PROVENANCE = "provenance"

    # The name of the dataset. This was added to enable having multiple dataset in a
    # single DataFrame while merging test positivity data sources.
    DATASET = "dataset"

    DEMOGRAPHIC_BUCKET = "demographic_bucket"

    # Name of distribution such as "age" or "race" or "age;race"
    DISTRIBUTION = "distribution"


# CommonFields used as keys/index columns in timeseries DataFrames.
# I'd like this to be immutable (for example a tuple) but pandas sometimes treats tuples and lists
# differently and many covid-data-model tests fail when it is a tuple.
COMMON_FIELDS_TIMESERIES_KEYS = [CommonFields.FIPS, CommonFields.DATE]


# Fields that are currently expected when representing a region in a DataFrame and CSV. Newer code is expected
# to only depend on the character field FIPS.
COMMON_LEGACY_REGION_FIELDS = [
    CommonFields.FIPS,
    CommonFields.STATE,
    CommonFields.COUNTRY,
    CommonFields.COUNTY,
    CommonFields.AGGREGATE_LEVEL,
]


COMMON_FIELDS_ORDER_MAP = {common: i for i, common in enumerate(CommonFields)}


class FieldNameAndCommonField(FieldName):
    """Represents the original field/column name and CommonField it maps to or None if dropped."""

    def __new__(cls, field_name: str, common_field: Optional[CommonFields]):
        o = super().__new__(cls, field_name)
        o.common_field = common_field
        return o
