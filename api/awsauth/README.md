# API Authentication Docs

## Setup

We deploy the lambda functions using the [Serverless Framework](https://www.serverless.com/framework/docs/).

Install serverless:

```
npm install -g serverless
```


## Deploy

Currently we have a `dev` and `prod` stage.

There are two environment variables:
 * `EMAILS_ENABLED`: If set, will send email on registration. Should be disabled on dev unless explicitly testing.
 * `SENTRY_DSN`: Sentry DSN used to report sentry errors. This should be set on prod.

```
EMAILS_ENABLED=true SENTRY_DSN=<sentry dsn> sls deploy --stage {dev,prod}
```
