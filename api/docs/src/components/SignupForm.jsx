import React, { useState } from "react";
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';

import {
  InputHolder,
  StyledNewsletter,
  GettingStartedBox,
  ApiKey,
  InputError,
} from "@site/src/components/SignupForm.style";

// Taken from https://ui.dev/validate-email-address-javascript/
const EMAIL_REGEX = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

const HUBSPOT_COOKIE_NAME = "hubspotutk";

// Taken from
// https://gist.github.com/joduplessis/7b3b4340353760e945f972a69e855d11
function getCookie(name) {
  const value = "; " + document.cookie;
  const parts = value.split("; " + name + "=");

  if (parts.length == 2) {
    return parts.pop().split(";").shift();
  }
}

const trackEmailSignupSuccess = (isNewUser) => {
  ga('send', {
    hitType: 'event',
    eventCategory: 'API Register',
    eventAction: 'Submit',
    eventLabel: isNewUser ? 'New User' : 'Existing User'
  });
  if (isNewUser) {
    gtag_report_conversion();
  }
};

const SignupForm = () => {
  const [email, setEmail] = useState();
  const [apiKey, setApiKey] = useState("");
  const [errorMessage, setErrorMessage] = useState("");
  const { siteConfig } = useDocusaurusContext();

  const onSubmit = async (e) => {
    e.preventDefault();
    setApiKey("");
    if (!EMAIL_REGEX.test(email)) {
      setErrorMessage("Must supply a valid email address");
      return;
    }

    const hubspotToken = getCookie(HUBSPOT_COOKIE_NAME);

    fetch(siteConfig.customFields.registerUrl, {
      method: "POST",
      body: JSON.stringify({
        email,
        hubspot_token: hubspotToken,
        page_uri: window.location.href,
      }),
      headers: { "Content-Type": "application/json" },
    })
      .then((res) => res.json())
      .then((data) => {
        setErrorMessage(undefined);
        // Older API returned data json encoded in "body" parameter.
        setApiKey(data.api_key);
        trackEmailSignupSuccess(data.new_user);
      })
      .catch((err) => setErrorMessage("Must supply a valid email address"));
  };

  return (
    <GettingStartedBox>
      <p>There are just two fast and simple steps before using our API.</p>

      <div>
        <h3>1. Get your API key</h3>
        <StyledNewsletter>
          <form>
            <InputHolder>
              <input
                autoComplete="Email"
                aria-label="Email"
                placeholder="Enter your email address"
                id="fieldEmail"
                maxLength="200"
                type="email"
                onChange={(e) => setEmail(e.target.value)}
              />
              <button type="submit" onClick={(e) => onSubmit(e)}>
                Get API key
              </button>
            </InputHolder>
            {errorMessage && <InputError>{errorMessage}</InputError>}
          </form>
        </StyledNewsletter>
        {!apiKey && (
          <p>
            If you've previously registered for an API key, you can use the form
            above to retrieve it.
          </p>
        )}
        {apiKey && (
          <p>
            Congrats, your new API key is <ApiKey>{apiKey}</ApiKey>
          </p>
        )}
      </div>
      <h3>
        2. Complete the{" "}
        <a
          href="https://docs.google.com/forms/d/e/1FAIpQLSf15Qx2EdYUHUmNI2JBts4LbVqIxsLN1SEzZLJlwuWdfJ4dVg/viewform?usp=sf_link"
          target="_blank"
          rel="noopener noreferrer"
        >
          registration form
        </a>
      </h3>
      <p>This helps establish a relationship between us for better support.</p>
    </GettingStartedBox>
  );
};

export default SignupForm;
