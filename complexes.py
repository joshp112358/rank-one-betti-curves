# Wrappers for functions that are used in the Jupyter Notebook figures.ipynb.
# Made by Carina Curto, Joshua Paik and Igor Rivin
from scipy.stats import rankdata
import numpy as np
from numpy import inf
import ripser
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics.pairwise import pairwise_distances
from scipy import sparse
import pandas as pd
import networkx as nx
import random
from ripser import ripser
from numpy.linalg import svd, matrix_rank

def submatrix(M,idx):
    return M[np.ix_(idx,idx)]

def plot_normalized_svd(M):
    A = M - np.mean(M)
    one_to_n = list(range(1,len(A)+1))
    [u,s,v] = svd(A)
    plt.plot(one_to_n, s)
    rrank = matrix_rank(M)
    plt.title("Rank = " + str(rrank),fontsize= 16)
    plt.xlabel("index of singular values",fontsize= 16)
    aspect_ratio = len(M)/np.max(s)
    plt.gca().set_aspect(aspect = aspect_ratio, adjustable='box')
    plt.ylabel("singular values",fontsize= 16)
    plt.xticks(fontsize= 18)
    plt.yticks(fontsize= 18)

def get_upper_triangle(M):
    n = len(M)
    if n <= 1:
        return "M is not a matrix of size greater than 0"
    output = []
    for i in range(0,n):
        for j in range(i+1,n):
            output.append(M[i,j])
    return output

def build_array_from_upper_triangle(array, n):
    M = np.zeros([n,n])
    index = 0
    for i in range(0,n):
        for j in range(i+1,n):
            M[i,j] = array[index]
            index += 1
    return M

def rankify(M):
    n = len(M)
    n_choose_2 = n*(n-1)/2
    upper_triangle_vector = get_upper_triangle(M)
    ranked_entries = rankdata(upper_triangle_vector, method = "min")
    output = build_array_from_upper_triangle(ranked_entries, n)
    output = output + output.T
    return output/n_choose_2

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

def all_betti_data(cluster, matrix):
    sub_mat = submatrix(matrix, cluster)
    rankified_sub_mat = rankify(-sub_mat)
    # Note that maxdim
    results = ripser(rankified_sub_mat, distance_matrix=True, maxdim=3, do_cocycles=True)
    betti0 = betti_data(results, 0)
    betti1 = betti_data(results, 1)
    betti2 = betti_data(results, 2)
    betti3 = betti_data(results, 3)

    for bc in [betti0, betti1, betti2, betti3]:
        bc[0] = np.array(list(bc[0]))
        bc[1] = np.array(list(bc[1]))
        #replace inf value
        bc[0][bc[0] == inf] = 1
        bc[0] = np.around(bc[0], 4)*10000
        bc[0] = bc[0].astype(int)


    return [betti0, betti1, betti2, betti3]

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
    plt.legend(prop={'size': 20})
    plt.xlabel("Edge Density", fontsize = 16)
    plt.xticks(fontsize= 18)
    plt.yticks(fontsize= 18)

def all_betti_plot(res):
    [B0, B1, B2, B3] = padded_betti_data(res)
    y_max = max([B0[1].max().max(),
                 B1[1].max().max(),
                 B2[1].max().max(),
                 B3[1].max().max()])

    betti_curves = [B0, B1, B2, B3]
    colors = [(.5, .5, .5), (.6, .5, .1), (.8, 0, 0),(.3, .3, 1)]
    for i in range(4):
        plot_betti_data(betti_curves[i], colors[i], i)
    plt.xlim([0, 1])
    plt.ylim([0, y_max+1])

def betti_plot(M):
    #input rankified M
    results = ripser(M, distance_matrix=True, maxdim=3, do_cocycles=True)
    return all_betti_plot(results)

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
    plt.legend(prop={'size': 20})

    y_max = max([dfB0.max().max(), dfB1.max().max(), dfB2.max().max(), dfB3.max().max()])
    plt.xlim([0, 1])
    plt.ylim([0, y_max+1])
    plt.xlabel("Edge Density")
    plt.xticks(fontsize= 18)
    plt.yticks(fontsize= 18)
