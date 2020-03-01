# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.

"""Utility object for easily computing running averages."""
from collections import defaultdict


class Meter(object):
    """Utility object for computing running averages."""

    def __init__(self, reset_on_value=False):
        """
        Initialization.

        Args:
            reset_on_value (bool): If true, resets the running average after
                                   `.value` is accessed.
        """
        self.reset_on_value = reset_on_value
        self.reset()

    def add_sample(self, val):
        """
        Adds the specified value to the running average.

        Args:
            val (float): The new value.
        """
        self.sample_vals += val
        self.sample_count += 1

    def value(self):
        """
        Returns the current running average.

        If `self.reset_on_value == True` then the running average will be reset
        after this call.

        NOTE: If no samples have been added, this returns `0`.
        """
        if self.sample_count == 0:
            return 0.0

        ret = self.sample_vals / self.sample_count

        if self.reset_on_value:
            self.reset()

        return ret

    def reset(self):
        """Resets the running average."""
        self.sample_vals = 0.0
        self.sample_count = 0


def MeterDict(reset_on_value=False):
    return defaultdict(lambda: Meter(reset_on_value=reset_on_value))
