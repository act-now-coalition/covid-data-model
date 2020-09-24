import styled from 'styled-components';


export const InputHolder = styled.div`
  align-items: baseline;
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

    input[type='email'] {
      flex: 3;
      display: block;
      padding: 0.25rem 0.75rem;
      line-height: 2rem;
      height: 3.5rem;
      outline: 0;
      border: 1px solid #CCCCCC;
      border-right: none;
      border-top-left-radius: 4px;
      border-bottom-left-radius: 4px;
      appearance: none;
      font-size: 0.875rem;
      box-sizing: border-box;
      font-family: 'Roboto', 'Helvetica', 'Arial', sans-serif;

      ::placeholder {
        color: #828282;
        font-size: 15px;
      }

      &:hover {
        border: 1px solid black;
        border-right: none;
      }

      &[hidden] {
        display: none;
      }
    }

    button[type='submit'] {
      cursor: pointer;
      display: block;
      appearance: none;
      box-sizing: border-box;
      height: 3.5rem;
      flex-shrink: 0;
      flex: 1;
      outline: 0;
      border-top-right-radius: 4px;
      border-bottom-right-radius: 4px;
      appearance: none;
      font-size: 0.875rem;
      padding: 0.25rem 1.25rem;
      line-height: 1rem;
      text-transform: uppercase;
      transition: 0.3s ease background-color;
      background-color: #3BBCE6;
      border: 1px solid #3BA5C8;
      color: #FFFFFF;
      font-weight: 700;
      font-family: 'Roboto', 'Helvetica', 'Arial', sans-serif;

      &:hover {
        background-color: #3BA5C8;
      }
    }
  }
`;
