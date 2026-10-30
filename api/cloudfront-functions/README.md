# CloudFront Functions (shutdown responses)

The Covid Act Now API is shut down. These two CloudFront Functions
(`cloudfront-js-2.0`, viewer-request) are all that still runs in front of it.
They live in AWS, not in a deploy pipeline, so these files are the record of
what is published. Keep them in sync with the `LIVE` stage.

| File | Distribution | Attached to | Behavior |
|---|---|---|---|
| `api-covidactnow-shutdown.js` | `api.covidactnow.org` (`E2ZCL2ZF6E5OL0`) | Default behavior, the only one, all methods | 403 JSON shutdown message for every request |
| `data-covidactnow-shutdown.js` | `data.covidactnow.org` (`E1P2FZUDPJRWIO`) | Default behavior | 403 for bulk files and for everyone except the covidactnow.org website (matched by `Origin`/`Referer`) |

The API distribution has no read access to the `data.covidactnow.org` bucket.
The bucket policy grants only `E1P2FZUDPJRWIO`. So even if its function were
detached, it could not serve data.

## Updating a function

```bash
ETAG=$(aws cloudfront describe-function --name <name> --stage DEVELOPMENT --query ETag --output text)
ETAG=$(aws cloudfront update-function --name <name> --if-match $ETAG \
  --function-config 'Comment=,Runtime=cloudfront-js-2.0' \
  --function-code fileb://api/cloudfront-functions/<name>.js --query ETag --output text)
aws cloudfront test-function --name <name> --if-match $ETAG --stage DEVELOPMENT --event-object fileb://<event>.json
aws cloudfront publish-function --name <name> --if-match $ETAG
```

Run `test-function` before you publish. A function that throws returns errors
for every request on the distribution.
