# Setting up python environment covid-data-model

## Install Virtualenv

These instructions use `pyenv` to install python `3.7.7` and create a virtualenv.

1. Install `pyenv` and `pyenv-virtualenv`.

  On Mac, you can install through homebrew:

  ```
  $ brew update
  $ brew install pyenv pyenv-virtualenv
  ```
  Complete installation by modifying your dotfiles: [pyenv](https://github.com/pyenv/pyenv#basic-github-checkout)
  [pyenv-virtualenv](https://github.com/pyenv/pyenv-virtualenv#installing-with-homebrew-for-macos-users)

2. Install specified python version.

  ```
  $ pyenv install 3.7.7
  ```

3. Create virtualenv

  This example uses `covid-data-model` as the name of the virtualenv
  ```
  pyenv virtualenv 3.7.7 covid-data-model
  ```

  Optional: Add a `.python-version` file with the name of your virtualenv to the `covid-data-model/` root.
  This will automatically activate the virtualenv when you enter the directory.


## Install Requirements

Make sure you are in your virtualenv and run:

```
make setup-dev
```

or:

```
pip install -r requirements.txt -r requirements_test.txt
```


## Run Tests

We use [pytest](https://docs.pytest.org/en/latest/contents.html) as our test runner.

To run unit tests:
```
make test
```

To run linting:
```
make lint
```
