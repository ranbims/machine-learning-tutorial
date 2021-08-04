import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import linear_regression_model

def loadTrainingData() -> Tuple[List[float], List[float]]:
    X = []
    Y = []
    with open("lianjia_2021_8_3.txt") as file:
        lines = file.readlines()
        for line in lines:
            value = line.strip().split("\t")
            X.append(float(value[0]))
            Y.append(float(value[1]))
    return (X, Y)

def toNumpyArray(X: List[float], Y: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    return (np.array(X).reshape(len(X), 1), np.array(Y).reshape(len(Y), 1))

def plotData(X: List[float], Y: List[float]):
    XS, YS = zip(*sorted(zip(X, Y)))
    plt.plot(XS, YS, 'ro')
    plt.show()

def main():
    X_raw, Y_raw = loadTrainingData()
    X, Y = toNumpyArray(X_raw, Y_raw)
    plotData(X, Y)
    pass

def test_main():
    X_raw, Y_raw = loadTrainingData()
    X, Y = toNumpyArray(X_raw, Y_raw)
    model_parameters = linear_regression_model.learn(X, Y)
    print(model_parameters)

if __name__ == "__main__":
    main()