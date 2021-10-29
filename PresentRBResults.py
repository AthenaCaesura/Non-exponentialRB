# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 21:53:55 2021

@author: Athena
"""

from RB import rb_eval
import matplotlib.pyplot as plt

def plot_shot(maxm=100, sampleperiod = 1):
    evals = [rb_eval(num) for num in range(1, maxm, sampleperiod)]
    plt.plot(list(range(1, maxm, sampleperiod)), evals)
    plt.ylabel("Survival Probability")
    plt.xlabel("Gate Sequence Length")
    plt.show()
    