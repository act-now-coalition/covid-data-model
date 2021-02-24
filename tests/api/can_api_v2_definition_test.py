from api import can_api_v2_definition


def test_annotations_is_complete():
    metrics_names = list(can_api_v2_definition.Metrics.__fields__.keys())
    actuals_names = list(can_api_v2_definition.Actuals.__fields__.keys())
    # This test checks that metrics and actuals have annotations in a consistent order.
    expected_names = actuals_names + metrics_names

    annotations_names = list(can_api_v2_definition.Annotations.__fields__.keys())

    missing_names = [name for name in expected_names if name not in annotations_names]
    for name in missing_names:
        print(
            f"    {name}: Optional[FieldAnnotations] = pydantic.Field(None, "
            f'description="Annotations for {name}")'
        )

    assert annotations_names == expected_names, "Annotations out of order"
