import numpy as np


def minmax(signals):
    signals_minmax = []
    for s in signals:
        signals_minmax.append((s - np.min(s)) / (np.max(s) - np.min(s)))

    return signals_minmax
