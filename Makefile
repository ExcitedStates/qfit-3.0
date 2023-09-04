# XXX we should gradually re-enable some of these
PYLINT_DISABLE = C0103,C0114,C0200,C0206,C0209,C0411,C0415,R0902,R0903,R0904,R0912,R0913,R0914,R0915,R1702,R1714,R1728,R1732,W0401,W0511,W0611,W0612,W0632,W1202,W1203,W1404,W1514

install:
	pip3 install .

test-unit:
	py.test tests/unit/test_*.py

test-int:
	py.test --verbose -n 4 tests/test_*.py

test: test-unit test-int

pylint:
	set -o pipefail && \
	  pylint --errors-only --disable=E1101 src/qfit/ 2>&1 | \
	  (grep -v RuntimeWarning || true) 1>&2
	set -o pipefail && \
	  pylint --disable=$(PYLINT_DISABLE) tests/ 2>&1 | \
	  (grep -v RuntimeWarning || true) 1>&2

clean:
	rm -rf build dist
	rm -rf src/qfit.egg-info
