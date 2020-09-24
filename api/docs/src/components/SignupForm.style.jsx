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

  &:last-child {
    padding-top: ${({ errMessageOpen }) => (errMessageOpen ? '6px' : '30px')};
  }
`;