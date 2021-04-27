import enum

from covidactnow.datapublic import common_fields

from libs.datasets import taglib
from libs.qa import timeseries_stats
from tests import test_helpers


def test_check_str_enum_names_match_values():
    known_exceptions = (
        common_fields.CommonFields.ALL_BED_TYPICAL_OCCUPANCY_RATE,
        common_fields.CommonFields.ICU_TYPICAL_OCCUPANCY_RATE,
        common_fields.FieldGroup.HEALTHCARE_CAPACITY,
        taglib.TagField.TYPE,
    )
    cls: enum.Enum
    for cls in test_helpers.get_concrete_subclasses_not_in_tests(common_fields.ValueAsStrMixin):
        mismatches = []
        for val in cls:
            if val in known_exceptions:
                continue
            if val.name.lower() != val.value:
                mismatches.append(val)
        if mismatches:
            suggestion = "\n".join(f"    {v.name} = {repr(v.name.lower())}" for v in mismatches)
            print(f"fix for enum name and value mismatches in {cls}:\n{suggestion}")
        assert mismatches == []
