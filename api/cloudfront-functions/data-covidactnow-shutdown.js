function handler(event) {
    var request = event.request;
    var headers = request.headers;
    var uri = request.uri;

    var blockedFiles = ['counties.timeseries.json', 'counties.timeseries.csv'];
    for (var i = 0; i < blockedFiles.length; i++) {
        if (uri.indexOf(blockedFiles[i]) !== -1) {
            return {
                statusCode: 403,
                statusDescription: 'Forbidden',
                headers: {
                    'content-type': { value: 'application/json' },
                    'access-control-allow-origin': { value: '*' }
                },
                body: JSON.stringify({
                    error: "This bulk data file is no longer available for download. "
                         + "For a copy of this data, please contact api@covidactnow.org."
                })
            };
        }
    }

    var origin = (headers['origin'] && headers['origin'].value) || '';
    var referer = (headers['referer'] && headers['referer'].value) || '';

    var allowedOrigins = [
        'https://covidactnow.org',
        'https://www.covidactnow.org',
        'https://staging.covidactnow.org'
    ];

    var allowed = false;
    for (var i = 0; i < allowedOrigins.length; i++) {
        if (origin === allowedOrigins[i] || referer.indexOf(allowedOrigins[i] + '/') === 0) {
            allowed = true;
            break;
        }
    }

    if (allowed) {
        return request;
    }

    return {
        statusCode: 403,
        statusDescription: 'Forbidden',
        headers: {
            'content-type': { value: 'application/json' },
            'access-control-allow-origin': { value: '*' }
        },
        body: JSON.stringify({
            error: "The Covid Act Now data API (data.covidactnow.org) has been shut down. "
                 + "For assistance or bulk data requests, please contact api@covidactnow.org."
        })
    };
}