import numpy as np
from plotutils import PlotInstance
import matplotlib.pyplot as plt

def initialize_parameters(w: float, b: float) -> dict:
    return {"w": w, "b": b}

def predict(X: np.ndarray, model_parameters: dict) -> np.ndarray:
    w = model_parameters["w"]
    b = model_parameters["b"]
    Y = w * X + b
    return Y

def computeCost(A: np.ndarray, Y: np.ndarray) -> float:
    return np.sum(np.power(A - Y, 2)) / (2 * A.shape[0])

def updateParameters(parameters: dict, X: np.ndarray, A: np.ndarray, Y: np.ndarray, learning_rate: float):
    w = parameters["w"]
    b = parameters["b"]
    m = A.shape[0]
    w_new = w - np.sum((A - Y) * X) / m * learning_rate
    b_new = b - np.sum((A - Y)) / m * learning_rate
    parameters["w"] = w_new
    parameters["b"] = b_new

def learn(X: np.ndarray, Y: np.ndarray, learning_rate = 0.0001, iteration = 10000, plot = True) -> dict:
    w = 0
    b = 0
    parameters = initialize_parameters(w, b)
    if (plot):
        plotInstance = PlotInstance()
        plotInstance.start_ploting()
        plt.plot(X, Y, 'ro')
    for i in range(iteration):
        A = predict(X, parameters)
        cost = computeCost(A, Y)
        print(cost)
        updateParameters(parameters, X, A, Y, learning_rate)
        w = parameters["w"]
        b = parameters["b"]
        print("w = {}, b = {}".format(w, b))
        if (plot):
            plotInstance.plot(lambda x: x * w + b, 0, 150, 0, 1500, 'b')
    if (plot):
        plotInstance.finish_ploting()
    return parameters
