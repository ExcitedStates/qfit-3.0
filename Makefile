install:
	pip3 install .

test-unit:
	py.test tests/unit/test_*.py

test-int:
	py.test --verbose -n 4 tests/test_*.py

test: test-unit test-int

pylint:
	set -o pipefail && \
	  pylint --errors-only --disable=E1101 tests/ 2>&1 | \
	  (grep -v RuntimeWarning || true) 1>&2
	set -o pipefail && \
	  pylint --errors-only --disable=E1101 src/qfit/ 2>&1 | \
	  (grep -v RuntimeWarning || true) 1>&2

clean:
	rm -rf build dist
	rm -rf src/qfit.egg-info
