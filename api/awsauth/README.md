# API Authentication Docs

## Setup

We deploy the lambda functions using the [Serverless Framework](https://www.serverless.com/framework/docs/).

Install serverless:

```
npm install -g serverless
```


## Deploy

Currently we have a `dev` and `prod` stage.

Deploys are set up to look for environment variables in a `.env` file.

Create a .env file similar to this template:

```
# AWS Dynamo Table Name for API Keys
API_KEY_TABLE_NAME=

# If true, will send email on registration
EMAILS_ENABLED=

# Sentry DSN for reporting errors
SENTRY_DSN=

# Sentry environment to choose where to send notificaions.
# Should be either 'staging' or 'production'
SENTRY_ENVIRONMENT=

# Cloudfront distribution ID number for data deploy
CLOUDFRONT_DISTRIBUTION_ID=

# AWS Kinesis Firehose table name for metric collections
FIREHOSE_TABLE_NAME=

# Hubspot credentials
HUBSPOT_API_KEY=
HUBSPOT_ENABLED=

# Emails added to the blocklist will be blocked from successful API requests.
# Accepts JSON encoded lists, i.e.: ["email@domain.com"]
EMAIL_BLOCKLIST=[]
```

`pydantic` uses the `python-dotenv` package to instantiate an
[EnvConstants](https://github.com/covid-projections/covid-data-model/blob/master/api/awsauth/awsauth/config.py#L4)
object with the contents of the .env file.  Refer to the class for the most up to
date variables.

```
sls deploy --stage {dev,prod}
```
