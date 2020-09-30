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

const SignupForm = () => {
  const [email, setEmail] = useState();
  const [apiKey, setApiKey] = useState("");
  const [errorMessage, setErrorMessage] = useState("");
  const {siteConfig} = useDocusaurusContext();

  const onSubmit = async (e) => {
    e.preventDefault();
    setApiKey("");
    if (!EMAIL_REGEX.test(email)) {
      setErrorMessage("Must supply a valid email address");
      return;
    }
    fetch(siteConfig.customFields.registerUrl, {
      method: "POST",
      body: JSON.stringify({ email }),
      headers: { "Content-Type": "application/json" },
    })
      .then((res) => res.json())
      .then((data) => {
        setErrorMessage(undefined);
        setApiKey(JSON.parse(data.body).api_key);
      })
      .catch((err) => setErrorMessage("Must supply a valid email address"));
  };

  return (
    <GettingStartedBox>
      <h2>Getting Started</h2>
      <p>There are just two fast and simple steps before using our API.</p>

      <div>
        <h3>1. Request an API key</h3>
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
