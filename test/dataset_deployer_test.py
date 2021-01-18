from libs import dataset_deployer


def test_flatten_dict():

    nested = {"key1": 1, "key2": {"nested1": "a", "nested2": "b"}}

    results = dataset_deployer.flatten_dict(nested)

    expected = {"key1": 1, "key2.nested1": "a", "key2.nested2": "b"}
    assert results == expected


def test_write_nested_csv_with_skipped_keys(tmp_path):
    output_path = tmp_path / "output.csv"

    data = [{"foo": {"bar": 1, "baz": 2}, "bar": {"baz": 1}}]
    dataset_deployer.write_nested_csv(data, output_path, keys_to_skip=["foo.bar", "bar"])
    header = output_path.read_text().split("\n")[0]
    assert header == "foo.baz"
