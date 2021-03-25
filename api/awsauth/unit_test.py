from awsauth.config import Config


def test_aws_auth_config_loads():
    Config.init()
    assert Config.Constants.API_KEY_TABLE_NAME is not None
    assert 0
