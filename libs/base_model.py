import pydantic
import simplejson


def _nan_safe_json_dumps(*args, **kwargs):
    return simplejson.dumps(*args, **kwargs, ignore_nan=True)


class APIBaseModel(pydantic.BaseModel):
    """Base model for API output."""

    class Config:
        json_dumps = _nan_safe_json_dumps

        @staticmethod
        def schema_extra(schema, model):
            # Updating json schema output to respect optional typed fields.
            # Without this code, the schema output doesn't signify that a value
            # is nullable.
            # This behavior will not be changed until
            # Pydantic v2.0 it seems https://github.com/samuelcolvin/pydantic/issues/1270.
            for field_name, field in model.__fields__.items():

                # Checking for fields that allow none (essentially indicating that the
                # type is Optional[<type>]) and required.
                # This is possibly more stringent than necessary (field.required may not be
                # necessary), but since this code is fairly manual, this applies the
                # minimum changes necessary for our code.
                if field.allow_none and field.required:
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
