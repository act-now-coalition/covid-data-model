from libs.datasets import taglib
from tests import test_helpers
from libs.datasets.timeseries import TagType


def _get_subclass_tag_types(cls):
    """Returns a list of all TAG_TYPE subclasses of concrete subclasses of cls."""
    return [k.TAG_TYPE for k in test_helpers.get_concrete_subclasses_not_in_tests(cls)]


def test_all_tag_subclasses_accounted_for():
    """Checks that every subclass of TagInTimeseries has exactly one TagType enum value."""
    subclass_tag_types = _get_subclass_tag_types(taglib.TagInTimeseries)
    assert set(taglib.TagType) == set(subclass_tag_types)
    assert len(list(taglib.TagType)) == len(subclass_tag_types)


def test_annotation_tag_types():
    annotation_tag_types = _get_subclass_tag_types(taglib.AnnotationWithDate)
    assert sorted(annotation_tag_types) == sorted(
        [
            TagType.CUMULATIVE_LONG_TAIL_TRUNCATED,
            TagType.CUMULATIVE_TAIL_TRUNCATED,
            TagType.ZSCORE_OUTLIER,
        ]
    )


def test_tag_type_to_class():
    assert set(taglib.TAG_TYPE_TO_CLASS.keys()) == set(taglib.TagType)
    for tag_type, tag_type_class in taglib.TAG_TYPE_TO_CLASS.items():
        assert tag_type_class.TAG_TYPE is tag_type


def test_enum_names_match_values():
    test_helpers.assert_enum_names_match_values(taglib.TagType)
    test_helpers.assert_enum_names_match_values(taglib.TagField, exceptions={taglib.TagField.TYPE})


def test_load_known_issue_json():
    parsed_old = taglib.KnownIssue.make_instance(content='{"disclaimer":"a","date":"2021-05-14"}')
    expected = test_helpers.make_tag(taglib.TagType.KNOWN_ISSUE, public_note="a", date="2021-05-14")
    assert parsed_old == expected
    parsed_new = taglib.KnownIssue.make_instance(content='{"public_note":"a","date":"2021-05-14"}')
    assert parsed_new == expected


def test_load_known_issue_json_empty_disclaimer():
    parsed_old = taglib.KnownIssue.make_instance(content='{"disclaimer":"","date":"2021-05-14"}')
    expected = test_helpers.make_tag(taglib.TagType.KNOWN_ISSUE, public_note="", date="2021-05-14")
    assert parsed_old == expected
    parsed_new = taglib.KnownIssue.make_instance(content='{"public_note":"","date":"2021-05-14"}')
    assert parsed_new == expected
