'''
This class is defined as a data node, given the initial parameter information
'''
from scipy.special import gamma
import matplotlib.pyplot as plt
import numpy as np
import math


class Data():
    D = [1, 1]
    A = np.array([[2,-0.11],[-5,2.2]])
    B = np.array([[-1.6, -0.1],[-0.18,-2.4]])
    I = np.array([0, 0])
    Alpha = np.array([[0.02, 0],[0, 0.8]])
    Beita = np.array([[0.02, 0.01], [0, 0.9]])
    Tt = np.array([[0.1, 0],[0, 0.1]])
    Ss = np.array([[0.1, 0],[0, 0.1]])
    C = np.array([[0.1, 0],[0.1, 0]])
    V = np.array([1, 1])