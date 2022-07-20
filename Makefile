install:
	pip3 install .

test-unit:
	py.test tests/unit/test_*.py

test-int:
	py.test tests/test_*.py

test: test-unit test-int

pylint:
	pylint --errors-only src/qfit/

clean:
	rm -rf build dist
	rm -rf src/qfit.egg-info
