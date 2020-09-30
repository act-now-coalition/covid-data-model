from typing import Optional
import pydantic
import numpy as np
from libs import base_model


def test_custom_model_nans_serialize_properly():
    class InnerClass(pydantic.BaseModel):
        val: float

    class OuterClass(base_model.APIBaseModel):
        val: float
        inner: InnerClass

    data = OuterClass(val=np.nan, inner=InnerClass(val=np.nan))
    expected = '{"val": null, "inner": {"val": null}}'
    assert data.json() == expected


def test_optional_field_modifies_schema_properly():
    class Bar(pydantic.BaseModel):
        val: int

    class Foo(base_model.APIBaseModel):
        bar: Optional[float] = pydantic.Field(...)
        baz: float = pydantic.Field(...)
        bee: Optional[Bar] = pydantic.Field(...)

    results = Foo.schema()

    expected = {
        "title": "Foo",
        "description": "Base model for API output.",
        "type": "object",
        "properties": {
            "bar": {"title": "Bar", "anyOf": [{"type": "number"}, {"type": "null"}]},
            "baz": {"title": "Baz", "type": "number"},
            "bee": {"anyOf": [{"$ref": "#/definitions/Bar"}, {"type": "null"}]},
        },
        "required": ["bar", "baz", "bee"],
        "definitions": {
            "Bar": {
                "title": "Bar",
                "type": "object",
                "properties": {"val": {"title": "Val", "type": "integer"}},
                "required": ["val"],
            }
        },
    }
    assert results == expected
