import pydantic
import simplejson


def _nan_safe_json_dumps(*args, **kwargs):
    return simplejson.dumps(*args, **kwargs, ignore_nan=True)


class APIBaseModel(pydantic.BaseModel):
    """Base model for API output."""

    class Config:
        json_dumps = _nan_safe_json_dumps
