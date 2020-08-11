import pydantic
import api
from libs import update_readme_schemas


class TestReference(pydantic.BaseModel):
    name: str


class TestSchema(pydantic.BaseModel):
    reference: TestReference = pydantic.Field(..., description="Test Reference")


def test_schema_md_output():

    schema = pydantic.schema.schema([TestReference, TestSchema])

    test_schema = schema["definitions"]["TestSchema"]

    results = update_readme_schemas.schema_to_md(test_schema, schema)

    expected_output = """
### TestSchema


| name      | type                            | description    |
|-----------|---------------------------------|----------------|
| reference | [TestReference](#TestReference) | Test Reference |
"""
    assert results == expected_output


def test_can_schemas_successfully_generate():
    model_classes = api.find_public_model_classes()
    schema = pydantic.schema.schema(model_classes)
    generated_markdown = update_readme_schemas.generate_markdown_for_schema_definitions(schema)

    # Couple of checks to make sure generating sensible data.
    assert "### Actuals" in generated_markdown
    assert "List of [RegionSummary](#RegionSummary)" in generated_markdown
    # Make sure that all model classes are in output
    for model_class in model_classes:
        assert model_class.__name__ in generated_markdown
