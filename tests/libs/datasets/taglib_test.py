import abc
import inspect
from typing import Iterable
from typing import Type

from libs.datasets import taglib
from libs.datasets import timeseries


def _get_subclasses(cls) -> Iterable[Type]:
    """Yields all subclasses of `cls`."""
    # From https://stackoverflow.com/a/33607093
    for subclass in cls.__subclasses__():
        yield from _get_subclasses(subclass)
        yield subclass


def _get_subclass_tag_types(cls):
    """Returns a list of all TAG_TYPE subclasses of concrete subclasses of cls. By concrete I
    mean no abstract methods and not directly subclassing abc.ABC."""
    not_abstract_subclasses = [k for k in _get_subclasses(cls) if not inspect.isabstract(k)]
    not_abc = [k for k in not_abstract_subclasses if abc.ABC not in k.__bases__]
    return [k.TAG_TYPE for k in not_abc]


def test_all_tag_subclasses_accounted_for():
    """Checks that every subclass of TagInTimeseries has exactly one TagType enum value."""
    subclass_tag_types = _get_subclass_tag_types(taglib.TagInTimeseries)
    assert set(taglib.TagType) == set(subclass_tag_types)
    assert len(list(taglib.TagType)) == len(subclass_tag_types)


def test_annotation_tag_types():
    annotation_tag_types = _get_subclass_tag_types(taglib.AnnotationWithDate)
    assert sorted(annotation_tag_types) == sorted(timeseries.ANNOTATION_TAG_TYPES)


def test_tag_type_to_class():
    assert set(taglib.TAG_TYPE_TO_CLASS.keys()) == set(taglib.TagType)
    for tag_type, tag_type_class in taglib.TAG_TYPE_TO_CLASS.items():
        assert tag_type_class.TAG_TYPE is tag_type
