# API Authentication (Lambda@Edge)

## Current Status (February 2026)

The authenticated API at `api.covidactnow.org` remains active until **March 11, 2026**. After that date, the `check_api_key_edge` function automatically rejects all requests with a shutdown message.

Key behaviors:
- **Registration:** Permanently closed. All registration requests return 403.
- **Bulk file downloads:** `counties.timeseries.json` and `counties.timeseries.csv` are blocked with a 403 directing users to contact api@covidactnow.org.
- **Valid API keys:** Still accepted for all other endpoints until the shutdown date.
- **After March 11, 2026:** All requests are rejected regardless of API key.

## Architecture

The Lambda@Edge functions are attached to the `api.covidactnow.org` CloudFront distribution (`E2ZCL2ZF6E5OL0`) as viewer-request handlers:

- `check_api_key_edge` -- validates API keys on `v2/*` paths
- `register_edge` -- handles `v2/register` (permanently returns 403)

The distribution uses an Origin Access Control (OAC) to read from the `data.covidactnow.org` S3 bucket. The S3 bucket has Block Public Access enabled, so only this distribution can access the data.

The separate `data.covidactnow.org` distribution (`E1P2FZUDPJRWIO`) has a CloudFront Function that returns a 403 JSON error for all requests. Direct S3 access is also blocked.

## Setup

We deploy the lambda functions using the [Serverless Framework](https://www.serverless.com/framework/docs/) v2.72.3.

Install serverless:

```
npm install -g serverless@2.72.3
```

## Deploy

We have `dev` and `prod` stages deployed via GitHub Actions (`.github/workflows/deploy_aws_api_infrastructure_dev.yml`).

A .env file is used to configure environment variables for deploy. `pydantic` uses the `python-dotenv` package to instantiate an [EnvConstants](https://github.com/act-now-coalition/covid-data-model/blob/main/api/awsauth/awsauth/config.py#L4) object with the contents of the .env file. Refer to the class for the required variables.

## Deploying code to production

When code in `api/awsauth/` is merged to `main`, the deploy process is automatically kicked off. It will deploy to dev and then run tests. To deploy to prod, manually trigger the workflow with `deploy_to_prod` set to `true`. Prod deploy requires manual approval.

## Updating .env file

If you are making changes that require the .env file to be changed, follow these steps:

 1. Request access to the [API .env content doc](https://docs.google.com/document/d/1lHD2cG0FYKoOc5Q17ugg20m-WcucD7UfkiTSChpmUhQ)
 2. Update value for dev and prod
 3. Update GitHub Actions secrets (the `DOTENV` secret)
 4. Trigger deploy.
