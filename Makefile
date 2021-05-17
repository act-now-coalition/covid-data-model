.PHONY: setup-dev test unittest lint fmt

setup-dev: requirements.txt requirements_test.txt
	pip install --upgrade -r requirements.txt -r requirements_test.txt
	pre-commit install

unittest:
	pytest -n 1 tests/

unittest-not-slow:
	pytest -k 'not slow' -n 1 --durations=5 tests/

lint:
	pytest --pylint -m pylint --pylint-jobs=2 .

# Run unittests then linting
test: unittest lint

test-fast: lint unittest-not-slow

fmt:
	black .
