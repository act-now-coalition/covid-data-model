# Setting up dev environment for covid-data-model

## Cloning the repo.

git-lfs must be installed to checkout a copy of `covid-data-model`.

1. Install git-lfs
  On mac, `brew install git-lfs`

2. Clone the `covid-data-model` repo
    ```
    $ git clone git@github.com:covid-projections/covid-data-model.git
    ```
If you clone the repo before installing git-lfs run `git lfs pull`[*](https://github.com/git-lfs/git-lfs/issues/325#issuecomment-149713215) to fetch the large data files.

## Install Python 3.10 in a Virtualenv

You can use conda or pyenv.  Note that if you're using Apple Silicon (an M1 Mac), you may have better luck using anaconda. Either way you need to set up a python `3.10` environment. On Mac Silicon (M1/M2 chips) Python `3.10.13` may be specifically required.

### Conda
You can install miniconda python 3.10 from here
[https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)

Then use the following:
- `conda create python=3.10 -n covid-data-model`
- `conda activate covid-data-model`

### pyenv

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
  $ pyenv install 3.10
  ```

3. Create virtualenv

  This example uses `covid-data-model` as the name of the virtualenv
  ```
  pyenv virtualenv 3.10 covid-data-model
  ```

  Optional: Add a `.python-version` file with the name of your virtualenv to the `covid-data-model/` root.
  This will automatically activate the virtualenv when you enter the directory.

## Install Requirements and pre-commit

Make sure you are in your virtualenv and run:

```
make setup-dev
```

or manually run the commands in [our Makefile](https://github.com/covid-projections/covid-data-model/blob/main/Makefile).


## Auto-formatting

We use [black](https://github.com/psf/black) to automatically format python code.
`make setup-dev` installs `pre-commit`, which helps to keep this maintainable by automatically
reformating modified files on commit.


## Run Tests

We use [pytest](https://docs.pytest.org/) as our test runner.

To run lint and all tests:
```
make test
```

Some tests depend on a large dataset which takes about 25 seconds to load. These tests have a `@pytest.mark.slow` function decorator.

To run lint and the tests that are not marked as slow:
```
make test-fast
```

To run just linting:
```
make lint
```

## Sentry

The code is set up to report errors to Sentry. The gitub action pulls the sentry_dsn for the prod instance from secrets stored within github. It is possible to have Sentry run locally and report errors to the dev sentry instance by adding the following to your .env

```
export SENTRY_DSN=https://<GET_SENTRY_DSN_FOR_DEV_INSTANCE>.ingest.sentry.io/<DEV_INSTANCE>
```
