#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays as st_arrays

from qfit.utils.normalize_to_precision import normalize_to_precision


@given(
    a=st_arrays(
        dtype=float,
        shape=st.tuples(
            st.integers(min_value=1, max_value=10),
        ),
        elements=st.floats(
            min_value=0.0, max_value=np.finfo(float).max/10, allow_nan=False, allow_infinity=False
        ),
    ).filter(lambda x: not np.isclose(x.sum(), 0)),
    decimal_places=st.integers(min_value=1, max_value=8),
)
def test_normalize_to_precision(a: np.ndarray, decimal_places: int) -> None:
    a_rounded = normalize_to_precision(a, decimal_places)

    # Function should normalize array to sum to 1.
    assert np.isclose(a_rounded.sum(), 1.0)

    # Function should be within a decimal place of the rounded value.
    assert np.allclose(
        a_rounded,
        (a / a.sum()).round(decimal_places),
        atol=10**-decimal_places,
    )
