from scipy.special import gamma
import matplotlib.pyplot as plt
import numpy as np
import math

t = [n for n in range(0,100)]
x1 = np.sin(t)
x2 = np.cos(t) 

plt.xlabel('x1')
plt.ylabel('x2')
plt.plot(x1, x2) 
plt.show()