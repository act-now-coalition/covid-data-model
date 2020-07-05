import pydantic
import simplejson


def _nan_safe_json_dumps(*args, **kwargs):
    return simplejson.dumps(*args, **kwargs, ignore_nan=True)


class BaseModel(pydantic.BaseModel):
    class Config:
        json_dumps = _nan_safe_json_dumps
