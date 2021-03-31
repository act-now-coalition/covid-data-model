import React, { useState } from 'react';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import { Form, Field } from 'react-final-form';
import { TextField } from 'final-form-material-ui';
import {
  InputHolder,
  StyledNewsletter,
  GettingStartedBox,
  ApiKey,
  InputError,
} from './SignupForm.style';
import { Button, Grid, Typography } from '@material-ui/core';

// Taken from https://ui.dev/validate-email-address-javascript/
const EMAIL_REGEX = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

const HUBSPOT_COOKIE_NAME = 'hubspotutk';

// Taken from
// https://gist.github.com/joduplessis/7b3b4340353760e945f972a69e855d11
function getCookie(name: string) {
  const value = '; ' + document.cookie;
  const parts = value.split('; ' + name + '=');

  if (parts.length == 2) {
    return parts.pop()?.split(';').shift();
  }
}

const trackEmailSignupSuccess = (isNewUser: boolean) => {
  // @ts-ignore
  ga('send', {
    hitType: 'event',
    eventCategory: 'API Register',
    eventAction: 'Submit',
    eventLabel: isNewUser ? 'New User' : 'Existing User',
  });
  if (isNewUser) {
    // @ts-ignore
    gtag_report_conversion();
  }
};

interface FormData {
  email: string;
  useCase?: string;
}

const validate = (values: FormData) => {
  const errors: Any = {};

  if (!values.email) {
    errors.email = 'Required';
  } else if (!EMAIL_REGEX.test(values.email)) {
    errors.email = 'Must supply a valid email address';
  }

  console.log(errors);
  return errors;
};

const SignupForm = () => {
  const [apiKey, setApiKey] = useState('');
  const [errorMessage, setErrorMessage] = useState('');
  const { siteConfig } = useDocusaurusContext();

  const onSubmit = async (values: FormData) => {
    setApiKey('');

    const hubspotToken = getCookie(HUBSPOT_COOKIE_NAME);

    fetch(siteConfig.customFields.registerUrl, {
      method: 'POST',
      body: JSON.stringify({
        email: values.email,
        hubspot_token: hubspotToken,
        page_uri: window.location.href,
        use_case: values.useCase,
      }),
      headers: { 'Content-Type': 'application/json' },
    })
      .then(res => res.json())
      .then(data => {
        setErrorMessage(undefined);
        // Older API returned data json encoded in "body" parameter.
        setApiKey(data.api_key);
        trackEmailSignupSuccess(data.new_user);
      })
      .catch(err => {
        console.log(err);
        setErrorMessage('Must supply a valid email address');
      });
  };

  return (
    <div>
      <StyledNewsletter>
        <Form
          onSubmit={onSubmit}
          validate={validate}
          render={({ handleSubmit, submitting, pristine, values }) => (
            <form onSubmit={handleSubmit} noValidate>
              <Grid container spacing={2}>
                <Grid item xs={12}>
                  <Typography>
                    There are just 2 questions we’d liked answered, and then you
                    can immediately get started.{' '}
                  </Typography>
                </Grid>
                <Grid container item xs={12}>
                  <Grid item xs={12}>
                    <Typography gutterBottom>
                      <strong>Email address</strong>
                    </Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Field<string>
                      component={TextField}
                      name="email"
                      label="Email"
                      type="email"
                      required
                      fullWidth
                      variant="outlined"
                      aria-label="Email"
                    />
                  </Grid>
                </Grid>
                {/* <Grid container item xs={12}>
                    <Grid item xs={12}>
                      <h5>Organization name</h5>
                    </Grid>
                    <Grid item xs={6}>
                      <Field
                        autoComplete="Organization"
                        aria-label="Organization"
                        fullWidth
                        variant="outlined"
                        name="organization"
                        label="Organization name"
                        component={TextField}
                        required
                      />
                    </Grid>
                  </Grid> */}

                <Grid container item xs={12}>
                  <Grid item xs={12}>
                    <Typography gutterBottom>
                      <strong>How do you intend to use our data?</strong>
                    </Typography>
                  </Grid>
                  <Grid item xs={12}>
                    <span>It’s optional, but it’s helpful for us to know:</span>
                    <ul>
                      <li>
                        The data/metrics you’re interested in (e.g. vaccine
                        data, risk levels, cases, deaths, etc.)
                      </li>
                      <li>
                        How you will be using this data (e.g. internal dashboard
                        for reopening offices in the northwest, school project
                        analyzing nationwide county data, an app to track covid
                        risk for friends and family, etc.)
                      </li>
                      <li>
                        The locations you’d like to use this for (e.g. all 50
                        states, counties in the Florida panhandle, or just Cook
                        County, IL)
                      </li>
                    </ul>
                  </Grid>
                  <Grid item xs={12}>
                    <Field<string>
                      aria-label="How you are using the data"
                      placeholder="How are you using the data"
                      label="Use Case"
                      rows={5}
                      variant="outlined"
                      component={TextField}
                      name="useCase"
                      fullWidth
                      multiline
                    />
                  </Grid>
                </Grid>
                <Grid item xs={12}>
                  <Typography variant="body2">
                    By requesting an API key you agree to the terms of our{' '}
                    <a
                      href="https://apidocs.covidactnow.org/#license"
                      target="_blank"
                    >
                      License.
                    </a>
                  </Typography>
                </Grid>
                <Grid item xs={3}>
                  <Button
                    size="large"
                    variant="contained"
                    type="submit"
                    color="primary"
                    disabled={submitting}
                    disableElevation
                  >
                    Get API key
                  </Button>
                </Grid>
                <Grid item xs={12}>
                  {!apiKey && (
                    <p>
                      If you've previously registered for an API key, you can
                      use the form above to retrieve it.
                    </p>
                  )}
                  {apiKey && (
                    <p>
                      Congrats, your new API key is <ApiKey>{apiKey}</ApiKey>
                    </p>
                  )}
                </Grid>
              </Grid>

              {errorMessage && <InputError>{errorMessage}</InputError>}
            </form>
          )}
        />
      </StyledNewsletter>
    </div>
  );
};

export default SignupForm;
