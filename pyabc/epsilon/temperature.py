import numpy as np
import pandas as pd
from typing import Callable, Dict, List, Union

from .base import Epsilon
from ..distance import RET_SCALE_LIN, StochasticKernel


class Temperature(Epsilon):
    """
    A temperatur scheme handles the decrease of the temperatures employed
    by a :class:`pyabc.acceptor.StochasticAcceptor` overtime.

    Parameters
    ----------

    schemes: Union[Callable, List[Callable]]
        Temperature schemes of the form
        ``Callable[[dict, **kwargs], float]`` returning proposed
        temperatures for the next time point.
    aggregate_fun: Callable[List[int], int]
        The function to aggregate the schemes by, of the form
        ``Callable[List[float], float]``.
        Defaults to taking the minimum.
    initial_temperature: float
        The initial temperature. If None provided, an AcceptanceRateScheme
        is used.
    maximum_nr_populations: int
        The maximum number of iterations as passed to ABCSMC.
        May be inf.
    temperatures: Dict[int, float]
        The temperatures, format key:temperature.
    """

    def __init__(
            self,
            schemes: Union[Callable, List[Callable]] = None,
            aggregate_fun: Callable[List[int], int] = None,
            initial_temperature: float = None):
        if schemes is None:
            schemes = [AcceptanceRateScheme(), ExponentialDecayScheme()]
        self.schemes = schemes

        if aggregate_fun is None:
            aggregate_fun = lambda xs: np.min(xs)
        self.aggregate_fun = aggregate_fun

        self.initial_temperature = initial_temperature

        self.max_nr_populations = None
        self.temperatures = {}

    def initialize(self,
                   t: int,
                   get_weighted_distances: Callable[[], pd.DataFrame],
                   max_nr_populations: int):
        self.max_nr_populations = max_nr_populations
        self._update(t, get_weighted_distances, 1.0)

    def update(self,
               t: int,
               weighted_distances: pd.DataFrame,
               acceptance_rate: float):
        self._update(t, lambda: weighted_distances, acceptance_rate)

    def _update(self,
                t: int,
                get_weighted_distances: Callable[[], pd.DataFrame],
                acceptance_rate: float):
        """
        Compute the temperature for time `t`.
        """
        # update the temperature

        if t >= self.max_nr_populations - 1:
            # t is last time
            temperature = 1.0
        elif not self.temperatures and self.initial_temperature is not None:
            # take the initial temperature
            temperature = self.initial_temperature
        else:
            # evalute schemes
            temps = []
            for scheme in self.schemes:
                temp = scheme(
                    t=t,
                    get_weighted_distances=get_weighted_distances,
                    pdf_normalization
                    acceptance_rate=acceptance_rate)
                temps.append(temp)
            logger.debug(f"Proposed temperatures: {temps}.")

            # compute next temperature based on proposals and fallback
            # should not be higher than before
            fallback = self.temperatures[t - 1] \
                if t - 1 in self.temperatures
            proposed_value = self.aggregate_fun(temps)
            # also a value lower than 1.0 does not make sense
            temperature = max(min(proposed_value, fallback), 1.0)

        # record found value
        self.temperatures[t] = temperature

    def __call__(self,
                t: int) -> float:
        return self.temperatures[t]
