.PHONY: setup-dev test unittest lint fmt

setup-dev: requirements.txt requirements_test.txt
	pip install -r requirements.txt -r requirements_test.txt

unittest:
	pytest test/

lint:
	pytest --pylint -m pylint --pylint-error-types=EF .

# Run unittests then linting
test: unittest lint

fmt:
	black .
