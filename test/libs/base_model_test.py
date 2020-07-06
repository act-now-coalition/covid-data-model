import pydantic
import numpy as np
from libs import base_model


def test_custom_model_nans_serialize_properly():
    class InnerClass(pydantic.BaseModel):
        val: float

    class OuterClass(base_model.BaseModel):
        val: float
        inner: InnerClass

    data = OuterClass(val=np.nan, inner=InnerClass(val=np.nan))
    expected = '{"val": null, "inner": {"val": null}}'
    assert data.json() == expected
