from libs import dataset_deployer


def test_flatten_dict():

    nested = {"key1": 1, "key2": {"nested1": "a", "nested2": "b"}}

    results = dataset_deployer.flatten_dict(nested)

    expected = {"key1": 1, "key2.nested1": "a", "key2.nested2": "b"}
    assert results == expected


def test_write_nested_csv_with_specified_header(tmp_path):
    output_path = tmp_path / "output.csv"

    data = [{"foo": {"bar": 1, "baz": 2}, "bar": {"baz": 1}}]
    dataset_deployer.write_nested_csv(data, output_path, header=["foo.baz"])
    header = output_path.read_text().split("\n")[0]
    assert header == "foo.baz"


def test_write_nested_csv_without_header(tmp_path):
    output_path = tmp_path / "output.csv"

    data = [{"foo": {"bar": 1, "baz": 2}, "bar": {"baz": 1}}]
    dataset_deployer.write_nested_csv(data, output_path)
    header = output_path.read_text().split("\n")[0]
    assert header == "foo.bar,foo.baz,bar.baz"
