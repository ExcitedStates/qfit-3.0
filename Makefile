# we should keep adding to this as we identify things to fix from pylint-warn
PYLINT_ERROR_WARNINGS = W0212,W0611,W0612

install:
	pip3 install .

test-unit:
	py.test --verbose -n 4 --durations=10 tests/unit/test_*.py

test-int:
	py.test --verbose -n 4 --durations=20 tests/test_*.py

test-int-quick:
	py.test --verbose -n 4 --durations=20 -m "not slow" tests/test_*.py

test: test-unit test-int

pylint:
	set -o pipefail && \
	  pylint --disable=R,C,W --enable=$(PYLINT_ERROR_WARNINGS) src/qfit/ 2>&1 | \
	  (grep -v RuntimeWarning || true) 1>&2
	set -o pipefail && \
	  pylint --disable=R,C,W0511,W0612,W0613,W0632 tests/ 2>&1 | \
	  (grep -v RuntimeWarning || true) 1>&2

# This is too aggressive to fix everything right now
pylint-warn:
	set -o pipefail && \
	  pylint --disable=R,C src/qfit/ 2>&1 | \
	  (grep -v RuntimeWarning || true) 1>&2

clean:
	rm -rf build dist
	rm -rf src/qfit.egg-info
