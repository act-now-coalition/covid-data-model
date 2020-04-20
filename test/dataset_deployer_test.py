from libs import dataset_deployer


def test_flatten_dict():

    nested = {
        'key1': 1,
        'key2': {
            'nested1': 'a',
            'nested2': 'b'
        }
    }

    results = dataset_deployer.flatten_dict(nested)

    expected = {
        'key1': 1,
        'key2.nested1': 'a',
        'key2.nested2': 'b'
    }
    assert results == expected
