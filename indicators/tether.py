import vectorbt as vbt
from numba import njit
from vectorbt import _typing as tp
from vectorbt.generic import nb as generic_nb


@njit
def rolling_tether_nb(high: tp.Array2d, low: tp.Array2d, period: int) -> tp.Array2d:
    low_val = generic_nb.rolling_min_nb(low, period, minp=period)  # lowest within period
    high_val = generic_nb.rolling_max_nb(high, period, minp=period)  # highest within period
    return (high_val + low_val) / 2


Tether = vbt.IndicatorFactory(
    input_names=['high', 'low'],
    param_names=['period'],
    output_names=['tether'],
).from_apply_func(rolling_tether_nb, period=13)
