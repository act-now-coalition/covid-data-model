from api import can_api_v2_definition


def test_annotations_is_complete():
    actuals_names = list(can_api_v2_definition.Actuals.__fields__.keys())
    metrics_names = list(can_api_v2_definition.Metrics.__fields__.keys())
    # This test checks that metrics and actuals have annotations in a consistent order.
    # Names ending "Details" are the old annotation fields.
    expected_names = [
        name for name in actuals_names + metrics_names if not name.endswith("Details")
    ]

    annotations_names = list(can_api_v2_definition.Annotations.__fields__.keys())

    # Make a list to preserve order of expected_names.
    missing_names = [name for name in expected_names if name not in annotations_names]
    if missing_names:
        print("Copying the following into can_api_v2_definition.py class Annotations.")
        for name in missing_names:
            print(
                f"    {name}: Optional[FieldAnnotations] = pydantic.Field(None, "
                f'description="Annotations for {name}")'
            )

    assert annotations_names == expected_names, "Expected Annotations: actuals then metrics"
