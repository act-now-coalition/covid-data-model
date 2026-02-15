> [!CAUTION]
> **This project is archived and winding down.** Data snapshots are frozen as of May 2024. The data pipeline, API, and all associated infrastructure are being shut down. The authenticated API (`api.covidactnow.org`) will be permanently shut off on **March 11, 2026**. For questions or data requests, contact api@covidactnow.org.

# COVID-19 Data Pipeline (Archived)

COVID data pipeline / API that supported https://covidactnow.org/.

It ingested data scraped via https://github.com/covid-projections/can-scrapers, combined it, calculated metrics, and generated data files for the [Covid Act Now API](https://apidocs.covidactnow.org/) and [website](https://covidactnow.org/).

## Current Status (February 2026)

- **Data snapshots:** Frozen since May 2024. No new data is being generated.
- **Authenticated API (`api.covidactnow.org`):** Active until March 11, 2026. Existing API key holders can still access non-bulk data endpoints.
- **V1 data endpoint (`data.covidactnow.org`):** Shut down. Returns a 403 JSON error for all requests.
- **Direct S3 access:** Blocked. The S3 bucket has Block Public Access enabled.
- **Bulk downloads (`counties.timeseries.json`, `counties.timeseries.csv`):** Blocked on all endpoints. Contact api@covidactnow.org for bulk data requests.

## Infrastructure Changes (February 15, 2026)

The following changes were made to address excessive bandwidth costs from automated scrapers:

- **S3 bucket (`data.covidactnow.org`):** Block Public Access enabled, ACLs disabled, Origin Access Control (OAC) added for the authenticated API CloudFront distribution.
- **`data.covidactnow.org` CloudFront distribution (`E1P2FZUDPJRWIO`):** A CloudFront Function intercepts all requests and returns a 403 JSON error before reaching S3.
- **`api.covidactnow.org` CloudFront distribution (`E2ZCL2ZF6E5OL0`):** Lambda@Edge auth updated to block bulk file downloads (`counties.timeseries.json` and `.csv`).
- **GitHub Actions workflows disabled:** `deploy_api`, `label_api_snapshot`, `update_repo_datasets`, `CI_test_deploy_api`, and `update_api_usage` workflows have been disabled. Only `deploy_aws_api_infrastructure_dev` (for auth changes), `python_unit_tests`, and `deploy_docs` remain active.

## Repository Structure

This repository is preserved for reference. The key areas are:

- `api/awsauth/` -- Lambda@Edge authentication function (still active, deployed via Serverless Framework)
- `api/docs/` -- API documentation site ([apidocs.covidactnow.org](https://apidocs.covidactnow.org))
- `.github/workflows/` -- GitHub Actions (most disabled, see above)
- `libs/`, `pyseir/` -- Data pipeline code (no longer run)
- `data/` -- Static reference datasets

## Contact

For questions, data requests, or API assistance: api@covidactnow.org
