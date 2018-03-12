import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from Dot_Task.Analysis.load_data import load_threshold_data
from Dot_Task.Exp_Design.utils import get_difficulties

def plot_weibull(alpha, beta=3.5, chance = .5):
    x = np.linspace(0,alpha*3,100)
    y = chance + (1.0-chance)*(1-np.exp( -(x/alpha)**(beta) ))
    plt.plot(x,y)
    return alpha

def plot_threshold_run(subj_code, dim='motion'):
    taskinfo, df = load_threshold_data(subj_code, dim)
    