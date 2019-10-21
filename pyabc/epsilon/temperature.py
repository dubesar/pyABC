from .base import Epsilon
from ..distance import RET_SCALE_LIN, StochasticKernel


class Temperature(Epsilon):
    """
    A temperatur scheme handles the decrease of the temperatures employed
    by a :class:`pyabc.acceptor.StochasticAcceptor` overtime.
    """

    def initialize(self,
                   t: int,
                   get_weighted_distances: Callable[[], pd.DataFrame],
                   distance_function: Distance):
        if not isinstance(distance_function, StochasticKernel):
            raise AssertionError("A Temperature epsilon can only be used"
                                 "with a StochasticKernel comparator.")
        self.ret_scale = distance_function.ret_scale
            
