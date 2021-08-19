from typing import Any
import matplotlib.pyplot as plt
import numpy as np

class PlotInstance:

    current_curve: Any = None

    def start_ploting(self):
        plt.ion()
        plt.show()

    def finish_ploting(self):
        plt.ioff()
        plt.show()

    def plot(self, f: callable, startX: int, endX: int, startY: int, endY: int, format_string: str = ""):
        x = np.arange(startX, endX, 0.05)
        y = f(x)
        if (self.current_curve != None):
            self.current_curve.pop(0).remove()
        plt.axis([startX, endX, startY, endY])
        self.current_curve = plt.plot(x, y, format_string)
        plt.draw()
        plt.pause(.001)

def test():
    instance = PlotInstance()
    instance.start_ploting()
    instance.plot(lambda a : a * a, 0, 10, 0, 100)
    instance.plot(lambda a : a, 0, 10, 0, 10)
    instance.finish_ploting()

if __name__ == "__main__":
    test()