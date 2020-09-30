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
                    if "type" in schema["properties"][field_name]:
                        existing_schema_type = schema["properties"][field_name]["type"]
                        schema["properties"][field_name] = ["null", existing_schema_type]
                        # schema['properties']['nullable'] = True
                    # print(field_name, field)
                    # print(field.field_info)
                    # print(field.outer_type_)
                    # print(field.type_)
                    # print(field.allow_none)
                # print(vars(field))
                # print(field.type_)
                # print(dir(field.type_))
            # for field in model.fields:
            #     print(field)
            # print(model)
