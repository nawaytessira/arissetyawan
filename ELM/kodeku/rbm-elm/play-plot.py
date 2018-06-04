import numpy as np
import matplotlib.pyplot as plt

def plotsub(x, y, show=False):
    line, = plt.plot(x, y)
    if show:
        plt.show()
    return line

x = np.arange(0,10,1);
y = x*x   

plotsub(x, y)
plotsub(x*10, y, show=True)