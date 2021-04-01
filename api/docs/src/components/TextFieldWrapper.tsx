/// Copied from https://github.com/Deadly0/final-form-material-ui/blob/master/src/TextField.tsx
/// Directly importing from final-form-material-ui package caused a SSR build error in
/// docusaurus build.

import React from 'react';
import { FieldRenderProps } from 'react-final-form';
import TextField from '@material-ui/core/TextField';

const TextFieldWrapper = ({
  input: { name, onChange, value, ...restInput },
  meta,
  ...rest
}: FieldRenderProps<string, any>) => {
  const showError =
    ((meta.submitError && !meta.dirtySinceLastSubmit) || meta.error) &&
    meta.touched;

  return (
    <TextField
      {...rest}
      name={name}
      helperText={showError ? meta.error || meta.submitError : undefined}
      error={showError}
      inputProps={restInput}
      onChange={onChange}
      value={value}
    />
  );
};

export default TextFieldWrapper;
