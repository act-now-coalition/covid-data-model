import styled from 'styled-components';

export const InputHolder = styled.div`
  display: flex;
  flex-direction: row;
  padding-top: 0.5rem;
  label {
    margin: 0 0 8px 8px;
    color: #4f4f4f;
  }
`;

export const StyledNewsletter = styled.div`
  form {
    display: flex;
    flex-direction: column;

    // input[type='email'] {
    //   flex: 3;
    //   display: block;
    //   padding: 0.25rem 0.75rem;
    //   line-height: 2rem;
    //   height: 3.5rem;
    //   outline: 0;
    //   border: 1px solid #cccccc;
    //   border-right: none;
    //   border-top-left-radius: 4px;
    //   border-bottom-left-radius: 4px;
    //   appearance: none;
    //   font-size: 0.875rem;
    //   box-sizing: border-box;
    //   font-family: 'Roboto', 'Helvetica', 'Arial', sans-serif;

    //   ::placeholder {
    //     color: #828282;
    //     font-size: 15px;
    //   }

    //   &:hover {
    //     border: 1px solid black;
    //     border-right: none;
    //   }

    //   &[hidden] {
    //     display: none;
    //   }
    // }
  }
`;

export const GettingStartedBox = styled.div`
  padding: 12px;
  background-color: rgba(242, 242, 242, 0.25);
  border: 1px solid rgb(242, 242, 242);
  margin-bottom: 32px;

  // p {
  //   margin-top: 12px;
  //   color: rgb(130, 130, 130);
  // }

  h2 {
    font-style: normal;
    font-weight: 700;
    line-height: 15px;
    letter-spacing: 0em;
    text-align: left;
    text-transform: uppercase;
  }

  h3 {
    font-style: normal;
    font-weight: 700;
    line-height: 20px;
    letter-spacing: 0em;
    text-align: left;
  }
`;

export const ApiKey = styled.span`
  font-family: Roboto Mono, 'Courier New', Courier, monospace;
  font-size: 15px;
  font-style: normal;
  font-weight: 700;
  line-height: 21px;
  letter-spacing: 0em;
  text-align: left;
  color: rgb(255, 150, 0);
`;

export const InputError = styled.div`
  font-size: 12px;
  color: red;
  padding-left: 0.5em;
`;
