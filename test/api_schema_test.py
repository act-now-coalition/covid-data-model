import json
import api


def test_schemas_up_to_date():

    # Check to make sure that all public schemas are up to date
    # with stored schemas.
    schemas = api.load_public_schemas()
    for schema in schemas:
        schema_to_check = schema.schema()
        path = api.SCHEMAS_PATH / f"{schema.__name__}.json"
        existing = json.load(path.open())
        assert schema_to_check == existing
