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

# Sentry DSN for reporting errors
SENTRY_DSN=
# If true, will send email on registration
EMAILS_ENABLED=

# Cloudfront distribution ID number for data deploy
CLOUDFRONT_DISTRIBUTION_ID=
```

```
sls deploy --stage {dev,prod}
```
