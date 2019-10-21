from typing import Callable, List

from .base import Distance


RET_SCALE_LIN = "RET_SCALE_LIN"
RET_SCALE_LOG = "RET_SCALE_LOG"
RET_SCALES = [RET_SCALE_LIN, RET_SCALE_LOG]


class StochasticKernel(Distance):
    """
    A stochastic kernel assesses the similarity between observed and
    simulated summary statistics or data via a probability measure.

    .. note::
        The returned value cannot be interpreted as a distance function,
        but rather as an inverse distance, as it increases as the similarity
        between observed and simulated summary statistics increases.
        Thus, a StochasticKernel should only be used together with a
        StochasticAcceptor.

    Parameters
    -----------

    ret_scale: str, optional (default = RET_SCALE_LIN)
        The scale of the value returned in __call__:
        Given a proability density p(x,x_0), the returned value
        can be either p(x,x_0) or log(p(x,x_0)).
    keys: List[str], optional
        The keys of the summary statistics, specifying the order to be used.
    pdf_max: float, optional
        The maximum possible probability density function value.
        Defaults to None and is then computed as the density at (x_0, x_0),
        where x_0 denotes the observed summary statistics.
        Must be overridden if pdf_max is to be used in the analysis by the
        acceptor.
        This value should be on the scale specified by ret_scale already.
    """

    def __init__(
            self,
            ret_scale=RET_SCALE_LIN,
            keys=None,
            pdf_max=None):
        StochasticKernel.check_ret_scale(ret_scale)
        self.ret_scale = ret_scale
        self.keys = keys
        self.pdf_max = pdf_max

    def initialize(
            self,
            t: int,
            get_sum_stats: Callable[[], List[dict]],
            x_0: dict = None):
        """
        Remember the summary statistic keys in sorted order,
        if not set in __init__ already.
        """
        # initialize keys
        if self.keys is None:
            self.initialize_keys(x_0)

    @staticmethod
    def check_ret_scale(ret_scale):
        if ret_scale not in RET_SCALES:
            raise ValueError(
                f"The ret_scale {ret_scale} must be one of {RET_SCALES}.")

    def initialize_keys(self, x):
        self.keys = sorted(x)
