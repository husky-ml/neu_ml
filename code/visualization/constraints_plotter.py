import pdb

import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def constraints_plotter(data, constraints):
    """Plots data and corresponding constraints.

    The data is plotted as circles / spheres, and the constraints between a
    pair of points is indicated with a line segment (blue for 'must-link',
    and red for 'cannot-link')

    Parameters
    ----------
    data : array, shape ( num_instances, dim )
        The data to plot. 'n_instances' indicates the number of data points.
        'dim' indicates the dimension of the data. 

    constraints : networkx graph
        The constraints are encoded in a networkx graph, each node should
        indicate an instance index, and edges indicate constraints. Each
        edge should have a string attribute called 'constraint', which must
        either take a value of 'must_link' or 'cannot_link'.     
    """
    dim = data.shape[1]

    if dim > 3 or dim < 2:
        raise ValueError("Data must be two or three dimensional")

    if dim == 2:        
        # Create 2D plot
        plt.scatter(data[:,0], data[:,1], 50, [0.0, 1.0, 0.0], alpha=0.8)

        # Now draw the constraints
        num_constraints = len(constraints.edges())
        for i in np.arange(0, num_constraints):
            pt1 = constraints.edges()[i][0]
            pt2 = constraints.edges()[i][1]
            x1 = data[pt1, 0]
            x2 = data[pt2, 0]
            y1 = data[pt1, 1]
            y2 = data[pt2, 1]
            if constraints.edge[pt1][pt2]['constraint'] == 'must_link':
                plt.plot([x1, x2], [y1, y2], c='b')
            else:
                plt.plot([x1, x2], [y1, y2], c='r')
    
        plt.show()
    else:
        fig = plt.figure()
        ax1 = fig.add_subplot(111, projection='3d')
        col = ax1.scatter(data[:, 0], data[:, 1], data[:, 2], s=100, c='r')

        # Now draw the constraints
        num_constraints = len(constraints.edges())
        for i in np.arange(0, num_constraints):
            pt1 = constraints.edges()[i][0]
            pt2 = constraints.edges()[i][1]
            x1 = data[pt1, 0]
            x2 = data[pt2, 0]
            y1 = data[pt1, 1]
            y2 = data[pt2, 1]
            z1 = data[pt1, 2]
            z2 = data[pt2, 2]
            if constraints.edge[pt1][pt2]['constraint'] == 'must_link':
                ax1.plot([x1, x2], [y1, y2], [z1, z2], c='b')
            else:
                ax1.plot([x1, x2], [y1, y2], [z1, z2], c='r')
    
        plt.show()

def get_colors(n_colors):
    """
    Get a selection of colors for plotting

    Parameters
    ----------
    n_colors : int
        The number of unique colors to obtain

    Returns
    -------
    colors : list of length 'n_colors'
        Each element is a 3-element tuple indicating a unique color. Each
        element of a given tuple is in the interval [0.0, 1,0]
    """
    colors = []
    
    for i in np.arange(0, n_colors):
        if i == 0:
            colors.append([1.0, 0.0, 0.0])
        elif i == 1:
            colors.append([0.0, 1.0, 0.0])
        elif i == 2:
            colors.append([0.0, 0.0, 1.0])
        else:
            colors.append(np.random.rand(3))

    return colors
