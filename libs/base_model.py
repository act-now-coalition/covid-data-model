import pydantic
import simplejson
import ujson
import orjson


def _nan_safe_json_dumps(*args, **kwargs):
    return orjson.dumps(*args, option=orjson.OPT_SERIALIZE_NUMPY, **kwargs).decode("utf-8")
    return simplejson.dumps(*args, **kwargs, ignore_nan=True)


class APIBaseModel(pydantic.BaseModel):
    """Base model for API output."""

    class Config:
        json_dumps = _nan_safe_json_dumps

        # TODO(tom): Try fix all the errors when this is extra is `forbid`
        extra = pydantic.Extra.ignore

        @staticmethod
        def schema_extra(schema, model):
            # Updating json schema output to respect optional typed fields.
            # Without this code, the schema output doesn't signify that a value
            # is nullable.
            # This behavior will not be changed until
            # Pydantic v2.0 it seems https://github.com/samuelcolvin/pydantic/issues/1270.
            for field_name, field in model.__fields__.items():

                # Checking for fields that allow none (essentially indicating that the
                # type is Optional[<type>]).
                if field.allow_none:
                    existing_field = schema["properties"][field_name]
                    if "type" in existing_field:
                        # Removing existing type field and adding a "anyOf" indicates that
                        # the field may contain any of the following subschemas
                        # https://json-schema.org/understanding-json-schema/reference/combining.html
                        existing_type = existing_field.pop("type")
                        existing_field["anyOf"] = [
                            {"type": existing_type},
                            {"type": "null"},
                        ]
                    elif "$ref" in existing_field:
                        existing_type = existing_field.pop("$ref")
                        existing_field["anyOf"] = [
                            {"$ref": existing_type},
                            {"type": "null"},
                        ]
