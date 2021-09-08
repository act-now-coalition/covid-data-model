.PHONY: setup-dev test unittest lint fmt

setup-dev: requirements.txt requirements_test.txt
	pip install --upgrade -r requirements.txt -r requirements_test.txt
	pre-commit install

# TODO(michael): We used to pass `-n 2` to pytest in order to run tests in
# parallel via pytest-xdist but it started being unreliable in CI (best guess
# is workers were running out of memory and crashing?) so we no longer do.
unittest:
	pytest tests/

unittest-not-slow:
	pytest -k 'not slow' -n 2 --durations=5 tests/

lint:
	pytest --pylint -m pylint --pylint-jobs=2 .

# Run unittests then linting
test: unittest lint

test-fast: lint unittest-not-slow

fmt:
	black .
