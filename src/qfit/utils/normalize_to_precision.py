#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


def normalize_to_precision(arr: np.ndarray, decimal_places: int) -> np.ndarray:
    """Normalize an array of numbers, while rounding to a specified number of decimal places.

    :param arr: The array to normalize.
    :param decimal_places: The number of decimal places to round to.
    """
    if arr.size == 1:
        return np.array([1.0])

    # Using full floating points, normalize the array
    normalized_arr = arr / arr.sum()

    # Round the array
    rounded_arr = np.round(normalized_arr, decimal_places)

    # Find the sum of the rounding errors ε, and how many last_dp need to be added
    eps = normalized_arr.sum() - rounded_arr.sum()
    if eps > 0:
        last_dp = 10.0**-decimal_places
    else:
        last_dp = -(10.0**-decimal_places)

    n_eps: int = round(eps / last_dp, None)

    # Add to the biggest element of rounded_arr
    ordering = np.argsort(rounded_arr)[::-1]  # Descending order
    for i in range(n_eps):
        rounded_arr[ordering[i]] += last_dp

    return rounded_arr
