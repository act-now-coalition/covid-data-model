import React, { useState } from "react";
import { InputHolder } from "@site/src/components/SignupForm.style";

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
    <div>
      <h3>Getting Started</h3>
      <span>
        There are just two fast and simple steps before using our API.
      </span>

      <div>
        <h4>1. Request an API Key</h4>
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
              Sign up
            </button>
          </InputHolder>
        </form>
        {!apiKey && (
          <div>
            <span>
              If you've previously registered for an API key, you can use the
              form above to retrieve it.
            </span>
          </div>
        )}
        {apiKey && (
          <div>
            <span>Congrats, your new API Key is {apiKey}</span>
          </div>
        )}
      </div>
      <h4>
        2. Complete the{" "}
        <a
          href="https://docs.google.com/forms/d/e/1FAIpQLSf15Qx2EdYUHUmNI2JBts4LbVqIxsLN1SEzZLJlwuWdfJ4dVg/viewform?usp=sf_link"
          target="_blank"
        >
          Registration Form
        </a>
        <span>
          This helps establish a relationship between us for better support.
        </span>
      </h4>
    </div>
  );
};

export default SignupForm;
