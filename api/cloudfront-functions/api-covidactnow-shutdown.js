// CloudFront Function (cloudfront-js-2.0), viewer-request on the DEFAULT behavior
// of api.covidactnow.org (E2ZCL2ZF6E5OL0), which is the distribution's only
// behavior once the v2/register and v2/* behaviors are removed.
//
// Replaces the awsauth Lambda@Edge pair. Since the 2026-03-11 shutdown,
// check_api_key_edge returned this exact body for every v2/* request;
// register_edge returned a "no longer accepting new registrations" variant,
// which this supersedes.
//
// The swap must be ONE update-distribution call (drop the Lambda behaviors, attach
// this function): any window without it lets v2/* through to S3 via the OAC.
var BODY = JSON.stringify({
    error: "The Covid Act Now API has been permanently shut down. "
        + "Data snapshots are no longer being updated and the API is no longer serving requests. "
        + "If you need assistance, please reach out to api@covidactnow.org."
});

function handler(event) {
    return {
        statusCode: 403,
        statusDescription: 'Forbidden',
        headers: {
            'content-type': { value: 'application/json' },
            'access-control-allow-origin': { value: '*' }
        },
        body: BODY
    };
}
