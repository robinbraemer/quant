import numpy as np
import vectorbt as vbt
from numba import njit
from vectorbt import _typing as tp
from vectorbt.generic import nb as generic_nb


# @njit
# def rolling_demarker_1d_nb(high: tp.Array1d, low: tp.Array1d, period: int) -> tp.Tuple[tp.Array1d, tp.Array1d, tp.Array1d]:
#     demin = np.empty_like(low, dtype=np.float_)
#     demax = np.empty_like(high, dtype=np.float_)
#     for i in range(high.shape[0]):
#         if low[i] < low[i - 1]:
#             demin[i] = low[i - 1] - low[i]
#         else:
#             demin[i] = 0
#
#         if high[i] > high[i - 1]:
#             demax[i] = high[i] - high[i - 1]
#         else:
#             demax[i] = 0
#
#     demax_avg = generic_nb.rolling_mean_1d_nb(demax, period, minp=period)  # sma
#     demin_avg = generic_nb.rolling_mean_1d_nb(demin, period, minp=period)  # sma
#
#     return demax_avg / (demax_avg + demin_avg), demin_avg, demax_avg

@njit
def rolling_demarker_2d_nb(high: tp.Array2d, low: tp.Array2d, period: int) -> tp.Tuple[tp.Array2d, tp.Array2d, tp.Array2d]:
    demin = np.empty_like(low, dtype=np.float_)
    demax = np.empty_like(high, dtype=np.float_)
    for col in range(high.shape[1]):
        for i in range(high.shape[0]):
            if low[i, col] < low[i - 1, col]:
                demin[i, col] = low[i - 1, col] - low[i, col]
            else:
                demin[i, col] = 0

            if high[i, col] > high[i - 1, col]:
                demax[i, col] = high[i, col] - high[i - 1, col]
            else:
                demax[i, col] = 0

    demax_avg = generic_nb.rolling_mean_nb(demax, period, minp=period)  # sma
    demin_avg = generic_nb.rolling_mean_nb(demin, period, minp=period)  # sma

    return demax_avg / (demax_avg + demin_avg), demin_avg, demax_avg


# @njit
# def rolling_demarker_nb(high: tp.Array2d, low: tp.Array2d, period: int) -> tp.Tuple[tp.Array2d, tp.Array2d, tp.Array2d]:
#     """2-dim version of `rolling_demarker_1d_nb`."""
#     out = np.empty_like(high, dtype=np.float_)
#     for col in range(high.shape[1]):
#         print( high[:, col])
#         out[:, col] = rolling_demarker_1d_nb(high[:, col], low[:, col], period)
#     # return out
#     return out, out, out # TODO where is what?


DeMarkerOscillator = vbt.IndicatorFactory(
    input_names=['high', 'low'],
    param_names=['period'],
    output_names=['dem', 'demin_avg', 'demax_avg'],
).from_apply_func(rolling_demarker_2d_nb, period=13)
