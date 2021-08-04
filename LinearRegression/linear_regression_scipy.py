from typing import List
from scipy import stats

def learn(X: List[float], Y: List[float]) -> dict:
    slope, intercept, r, p, std_err = stats.linregress(X, Y)
    parameters = {}
    parameters['w'] = slope
    parameters['b'] = intercept
    return parameters
