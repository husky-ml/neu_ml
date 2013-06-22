import pdb

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from neu_ml.code.algorithms.regression.gp_regressors import *

def plot_gp_regressors(regressors, input_dims=None, target_dims=None,
                       axes=None):
    """Plots the specified Gaussian Process regressors with scatter-plots for
    the input-target pairs and line plots for the Gaussian Process regression
    curves.

    If the input and target dimensions are both equal to one, a 2D plot will be
    produced. If the input dimension is two and the target dimension is one, or
    if the input dimension is one and the target dimension is two, a 3D plot
    will be produced. Otherwise an error will be generated.

    It's assumed that each regressor has the same input and target dimension.

    Parameters
    ----------
    regressors : list of GPRegressors
        List of gp_regressor elements to be plotted

    input_dims : array, shape ( m ), optional
        An array of indices indicating which input dimensions to use for
        plotting. This array must have two elements or less in order to produce
        meaningful plots. If not specified, the first input dimension will be
        used for plotting.

    target_dims : array, shape ( n ), optional
        An array of indices indicating which target dimensions to use for
        plotting. This array must have two elements or less in order to produce
        meaningful plots. If not specified, the first target dimension will be
        used for plotting.

    axes : matplotlib.axes.Axes instance, optional
        The plot will be generated within this instance instead of plotted
        to directly to the screen        
    """
    n_regressors = len(regressors)
    if regressors[0].get_inputs().ndim == 1:
        input_dim = 1
    else:
        input_dim = regressors[0].get_inputs().shape[1]
    if regressors[0].get_targets().ndim == 1:
        target_dim = 1
    else:
        target_dim = regressors[0].get_targets().shape[1]

    # Determine the effective input dimensions ('i_dims') and the target
    # dimensions ('t_dims') to plot.
    i_dims = []
    if input_dim == 1:
        i_dims.append(0)
    elif input_dim == 2:
        i_dims.append(0)
        i_dims.append(1)
    elif input_dims is None:
        raise ValueError("Input dimension not specified")
    else:
        for i in np.arange(0,len(input_dims)):
            i_dims.append(input_dims[i])

    t_dims = []
    if target_dim == 1:
        t_dims.append(0)
    elif target_dim == 2:
        t_dims.append(0)
        t_dims.append(1)
    elif target_dims is None:
        raise ValueError("Target dimension not specified")
    else:
        for i in np.arange(0,len(target_dims)):
            t_dims.append(target_dims[i])

    if len(i_dims) + len(t_dims) > 3:
        raise ValueError("Can not plot more than three dimensions")

    if len(i_dims) == 1 and len(t_dims) == 1:
        plot_1d_input_1d_target(regressors, i_dims, t_dims, axes)
    elif len(i_dims) == 1 and len(t_dims) == 2:
        plot_1d_input_2d_target(regressors, i_dims, t_dims, axes)
    else:
        plot_2d_input_1d_target(regressors, i_dims, t_dims, axes)

def get_colors(n_colors):
    """Get a selection of colors for plotting

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

def plot_1d_input_1d_target(regressors, i_dims, t_dims, axes=None):
    """
    Parameters
    ----------
    regressors : list of GPRegressors
        List of gp_regressor elements to be plotted

    i_dims : list with one element
        A list containing the index of the input dimension to plot

    t_dims : list with one element
        A list containing the index of the target dimension to plot

    axes : matplotlib.axes.Axes instance, optional
        The plot will be generated within this instance instead of plotted
        to directly to the screen
    """
    n_regressors = len(regressors)
    
    # Get a collection of unique colors for plotting
    colors = get_colors(n_regressors)
    
    # Create the scatter plots for the input-target pairs of each of the
    # Gaussian process regressors. Also get the extent of the domain over
    # which we have observed input data.
    input_min = np.finfo('f').max
    input_max = np.finfo('f').min
        
    for k in np.arange(0, n_regressors):
        # Only a subset of the input-target pairs contained in the regressor
        # may be "active". Obtain the active inputs and targets for plotting
        inputs = regressors[k].get_active_inputs()
        targets = regressors[k].get_active_targets()
        
        if np.max(inputs[:]) > input_max:
            input_max = np.max(inputs[:])
        if np.min(inputs[:]) < input_min:
            input_min = np.min(inputs[:])                

        if axes is not None:
            axes.scatter(inputs, targets, 50, colors[k], alpha=0.8)
        else:
            plt.scatter(inputs, targets, 50, colors[k], alpha=0.8)

    # Get the domain points at which to evaluate predicted Gaussian Process
    # values
    n_pts = 200
    pts = np.linspace(input_min, input_max, n_pts)

    # Now get the predicted target values for each Gaussian Process regressor
    for k in np.arange(0, n_regressors):
        [means, vars] = regressors[k].get_target_predictions(pts)
        if axes is not None:
            axes.plot(pts, means, 'k')
        else:
            plt.plot(pts, means, 'k')    

    if axes is None:
        plt.show()

def plot_1d_input_2d_target(regressors, i_dims, t_dims, axes=None):
    """
    Parameters
    ----------
    regressors : list of GPRegressors
        List of gp_regressor elements to be plotted

    i_dims : list with one element
        A list containing the index of the input dimension to plot

    t_dims : list with two element
        A list containing the index of the target dimension to plot

    axes : matplotlib.axes.Axes instance, optional
        The plot will be generated within this instance instead of plotted
        to directly to the screen
    """
    n_regressors = len(regressors)

    # Set up the figure for plotting
    if axes is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    
    # Get a collection of unique colors for plotting
    colors = get_colors(n_regressors)
    
    # Create the scatter plots for the input-target pairs of each of the
    # Gaussian process regressors. Also get the extent of the domain over
    # which we have observed input data.
    input_min = np.finfo('f').max
    input_max = np.finfo('f').min
        
    for k in np.arange(0, n_regressors):
        # Only a subset of the input-target pairs contained in the regressor
        # may be "active". Obtain the active inputs and targets for plotting
        inputs = regressors[k].get_active_inputs()
        targets = regressors[k].get_active_targets()
        
        if np.max(inputs[:]) > input_max:
            input_max = np.max(inputs[:])
        if np.min(inputs[:]) < input_min:
            input_min = np.min(inputs[:])                

        if axes is not None:
            axes.scatter(targets[:, 0], targets[:, 1], inputs[:, 0],
                         c=colors[k], alpha=0.8)
        else:
            ax.scatter(targets[:, 0], targets[:, 1], inputs[:, 0],
                         c=colors[k], alpha=0.8)

    # Get the domain points at which to evaluate predicted Gaussian Process
    # values
    n_pts = 500
    pts = np.atleast_2d(np.linspace(input_min, input_max, n_pts)).T

    # Now get the predicted target values for each Gaussian Process regressor
    for k in np.arange(0, n_regressors):
        [means, vars] = regressors[k].get_target_predictions(pts)

        if axes is not None:
            axes.plot(means[:, 0], means[:, 1], pts[:, 0], c='k')
        else:
            ax.plot(means[:, 0], means[:, 1], pts[:, 0], c='k')

    if axes is None:
        plt.show()

def plot_2d_input_1d_target(regressors, i_dims, t_dims, axes=None):
    """
    Parameters
    ----------
    regressors : list of GPRegressors
        List of gp_regressor elements to be plotted

    i_dims : list with one element
        A list containing the index of the input dimension to plot

    t_dims : list with two element
        A list containing the index of the target dimension to plot

    axes : matplotlib.axes.Axes instance, optional
        The plot will be generated within this instance instead of plotted
        to directly to the screen

    """
    pass 
