# Setting up dev environment for covid-data-model

## Copy the source data

Copy the source data from the `covid-data-public` repo to a sibling of your local `covid-data-model` directory. git-lfs must be
installed to checkout a copy of `covid-data-public`.

1. Install git-lfs
  On mac, `brew install git-lfs`

2. Clone the `covid-data-public` repo
  ```
  $ cd ..
  $ git clone git@github.com:covid-projections/covid-data-public.git
  $ cd -

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

## Install Requirements and pre-commit

Make sure you are in your virtualenv and run:

```
make setup-dev
```

or manually run the commands in [our Makefile](https://github.com/covid-projections/covid-data-model/blob/master/Makefile).


### Auto-formatting

We use [black](https://github.com/psf/black) to automatically format python code.
`make setup-dev` installs `pre-commit`, which helps to keep this maintainable by automatically
reformating modified files on commit.


## Configuration

If you've cloned the covid-data-public repo to locally, set the environment variable to enable caching
```bash
export COVID_DATA_PUBLIC=../covid-data-public
```

In addition, some models may require a results/test directory ahead of time, so initialize
```bash
mkdir -p results/test/
```

## Run Just Api Generateion
Grab the results of a snapshot's model output (either from the s3 bucket or from a githubaction artifacts). 
Move those files to results/ folder (or folder of your choice)

```bash
  ./run.py deploy-states-api -i "results/" -o "output/" --summary-output "output/"
```


## Run Tests

We use [pytest](https://docs.pytest.org/en/latest/contents.html) as our test runner.

To run all tests:
```
make test
```

To run just unittests:
```
make unittest
```

To run just linting:
```
make lint
```
