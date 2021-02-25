# API Authentication Docs

## Setup

We deploy the lambda functions using the [Serverless Framework](https://www.serverless.com/framework/docs/).

Install serverless:

```
npm install -g serverless
```


## Deploy

Currently we have a `dev` and `prod` stage (Serverless Framework uses "stages" as "environments") deployed via Github Actions.

A .env file is used to configure most of the environment variables for deploy.

`pydantic` uses the `python-dotenv` package to instantiate an
[EnvConstants](https://github.com/covid-projections/covid-data-model/blob/main/api/awsauth/awsauth/config.py#L4)
object with the contents of the .env file.  Refer to the class for the most up to
date variables.

## Deploying code to production

When code in api/awsauth is merged, the deploy process is automatically kicked off.
It will deploy to dev and then run tests.  Once the tests succeed, the deploy waits for a
manual approval to deploy to prod. This is limited right now to make sure that someone is around to
help monitor the API after it's deployed.


## Updating .env file

If you are making changes that require the .env file to be chnaged, follow these steps:

 1. Request access to the [API .env content doc](https://docs.google.com/document/d/1lHD2cG0FYKoOc5Q17ugg20m-WcucD7UfkiTSChpmUhQ)
 2. Update value for dev and prod
 3. Update Github action secrets
 4. Trigger deploy.
