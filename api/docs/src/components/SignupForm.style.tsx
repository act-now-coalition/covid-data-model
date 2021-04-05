import { createMuiTheme } from '@material-ui/core';
import styled from 'styled-components';

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

export const signupFormTheme = createMuiTheme({
  palette: {
    primary: {
      light: 'rgba(53, 103, 253, .15)',
      main: '#3567FD',
      dark: '#002CB4',
    },
  },
});
