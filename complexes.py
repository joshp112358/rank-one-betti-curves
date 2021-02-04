# Wrappers for functions that are used in the Jupyter Notebook figures.ipynb.
# Made by Carina Curto, Joshua Paik and Igor Rivin
from scipy.stats import rankdata
import numpy as np
import ripser
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics.pairwise import pairwise_distances
from scipy import sparse
import pandas as pd
import networkx as nx
import random
from ripser import ripser

def rankify(ar):
    n = ar.shape[0]
    mm= np.min(np.min(ar))
    out = np.copy(ar)
    np.fill_diagonal(out, mm-1)
    tmp=(rankdata(out, method='min')-1)/(n*n)
    return tmp.reshape(n, n)

def random_rank_k_mat(k, n):
    x = np.random.rand(k, n)
    xx = x.T @ x
    return xx

def mixed_sign_rank_k_mat(k,n):
    x = np.random.uniform(low=-1, high=1.0, size=(k,n))
    xx = x.T @ x
    return xx

def uniform_random_symmetric(n):
    x = np.random.uniform(low=0, high=1.0, size=(n,n))

    return 0.5*(x+x.T)

def therank(res, dim, t):
    thelist = res['dgms'][dim]
    if thelist.shape[0]==0:
        return 0
    births = thelist[:, 0]
    deaths = thelist[:, 1]
    howmanyb = births[births<=t].shape[0]
    howmanyd = deaths[deaths<=t].shape[0]
    return howmanyb - howmanyd

def theranks(res, dim):
    thelist = res['dgms'][dim]
    tlist = np.unique(thelist.flatten())
    return [(t, therank(res, dim, t)) for t in tlist]

# the rank and the ranks compute homology

def pad_data(array):
    # this function pads the ranks data to make it more uniform
    current_value = "filler"
    next_value = "filler2"
    result = array.copy()
    first = True
    for i in range(1,len(array)):
        current_value = result[i-1]
        next_value = result[i]
        if first:
            if current_value == -1:
                result[i-1] = 0
            if next_value != -1:
                first = False

        if next_value == -1:
            result[i] = current_value
    return result

def betti_data(res, dim):
    r1 = theranks(res, dim)
    if len(r1)>0:
        x, y = zip(*r1)
    else:
        x, y = (np.arange(0, 1.0, 0.01), np.zeros(100))
    return [x,y]

def padded_betti_data(res):
    # makes padded betti data up to dimension 3
    betti0 = betti_data(res, 0)
    betti1 = betti_data(res, 1)
    betti2 = betti_data(res, 2)
    betti3 = betti_data(res, 3)

    for bc in [betti0, betti1, betti2, betti3]:
        bc[0] = np.array(list(bc[0]))
        bc[1] = np.array(list(bc[1]))
        #replace inf value
        bc[0][bc[0] == np.inf] = 1
        bc[0] = np.around(bc[0], 4)*10000
        bc[0] = bc[0].astype(int)

        # Now pad the vector
        dummy_vector = -1*np.ones(10001)
        dummy_vector[bc[0]] = bc[1]
        dummy_vector = pad_data(dummy_vector)

        bc[0] = np.arange(0,1.0001, 0.0001)
        bc[1] = dummy_vector

    return [betti0, betti1, betti2, betti3]

def plot_betti_data(B, color, dim):
    x = B[0]
    y = B[1]
    lab=r"$\beta_{{{}}}$".format(dim)
    if dim == 0:
         plt.plot(x, y, label=lab, linestyle = '--', color = color)
    else:
        plt.plot(x, y, label=lab, color = color)
    plt.legend()

def all_betti_plot(res):
    [B0, B1, B2, B3] = padded_betti_data(res)
    betti_curves = [B0, B1, B2, B3]
    colors = [(.5, .5, .5), (.6, .5, .1), (.8, 0, 0),(.3, .3, 1)]
    for i in range(4):
        plot_betti_data(betti_curves[i], colors[i], i)

def plot_from_dataframes(dfB0, dfB1, dfB2, dfB3, alpha, linestyle):
    colors = [(.5, .5, .5), (.6, .5, .1), (.8, 0, 0),(.3, .3, 1)]
    num_of_rows = len(dfB0)
    x = np.arange(0,1.0001,0.0001)

    for i in range(num_of_rows):
        Betti_Curves = [dfB0.loc[i], dfB1.loc[i], dfB2.loc[i], dfB3.loc[i]]
        for dim in range(4):
            lab=r"$\beta_{{{}}}$".format(dim)
            y = Betti_Curves[dim]
            if dim == 0:
                plt.plot(x, y,
                         linestyle = "solid",
                         color = colors[dim],
                         alpha = alpha)
            else:
                plt.plot(x, y,
                         linestyle = "solid",
                         color = colors[dim],
                         alpha = alpha)

    #plot average

    average_B0 = dfB0.mean(axis = 0)
    average_B1 = dfB1.mean(axis = 0)
    average_B2 = dfB2.mean(axis = 0)
    average_B3 = dfB3.mean(axis = 0)

    Average_Betti_Curves = [average_B0,average_B1,average_B2,average_B3]
    for dim in range(4):
        lab=r"$\beta_{{{}}}$".format(dim)
        y = Average_Betti_Curves[dim]
        if dim == 0:
            plt.plot(x, y,
                     label=lab,
                     linestyle = linestyle,
                     color = colors[dim],
                     alpha = 1)
        else:
            plt.plot(x, y,
                     label=lab,
                     linestyle = linestyle,
                     color = colors[dim],
                     alpha = 1)
    plt.legend()

    y_max = max([dfB0.max().max(), dfB1.max().max(), dfB2.max().max(), dfB3.max().max()])
    plt.xlim([0, 1])
    plt.ylim([0, y_max+1])
