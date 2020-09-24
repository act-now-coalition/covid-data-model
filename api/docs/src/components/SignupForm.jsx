import React, { useState } from "react";
import {
  InputHolder,
  StyledNewsletter,
  GettingStartedBox,
  ApiKey,
} from "@site/src/components/SignupForm.style";

const SignupForm = () => {
  const [email, setEmail] = useState();
  const [apiKey, setApiKey] = useState("");

  console.log(apiKey);
  const onSubmit = async (e) => {
    e.preventDefault();
    fetch("https://api-dev.covidactnow.org/v2/register", {
      method: "POST",
      body: JSON.stringify({ email }),
      headers: { "Content-Type": "application/json" },
    })
      .then((res) => res.json())
      .then((data) => {
        console.log(data.body);
        setApiKey(JSON.parse(data.body).api_key);
      })
      .catch((err) => console.log(`Error: ${err}`));
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
                //ref={(i) => (this.emailInput = i)}
                autoComplete="Email"
                aria-label="Email"
                placeholder="Enter your email address"
                className="js-cm-email-input qa-input-email"
                id="fieldEmail"
                maxLength="200"
                // name="cm-yddtsd-yddtsd"
                required=""
                type="email"
                onChange={(e) => setEmail(e.target.value)}
              />
              <button type="submit" onClick={(e) => onSubmit(e)}>
                Get API key
              </button>
            </InputHolder>
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
