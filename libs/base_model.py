import pydantic
import simplejson


def _nan_safe_json_dumps(*args, **kwargs):
    return simplejson.dumps(*args, **kwargs, ignore_nan=True)


class APIBaseModel(pydantic.BaseModel):
    """Base model for API output."""

    class Config:
        json_dumps = _nan_safe_json_dumps

        def schema_extra(schema, model):
            print(schema)
            for field_name, field in model.__fields__.items():
                if field.allow_none and field.required:
                    print("*****")
                    print(field_name)
                    existing_field = schema["properties"][field_name]
                    if "type" in existing_field:
                        existing_type = existing_field.pop("type")
                        existing_field["anyOf"] = [
                            {"type": existing_type},
                            {"type": "null"},
                        ]
                    if "$ref" in existing_field:
                        existing_type = existing_field.pop("$ref")
                        existing_field["anyOf"] = [
                            {"$ref": existing_type},
                            {"type": "null"},
                        ]
