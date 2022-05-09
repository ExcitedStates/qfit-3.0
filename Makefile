install:
	pip3 install .

test-unit:
	py.test tests/unit/test_*.py

pylint:
	pylint --errors-only src/qfit/

clean:
	rm -rf build dist
	rm -rf src/qfit.egg-info
