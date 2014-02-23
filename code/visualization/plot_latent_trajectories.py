import pdb

import numpy as np
from numpy import dot, max, min
from scipy.linalg import solve
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from neu_ml.code.utils.compute_gram_mat import compute_gram_mat
from neu_ml.code.algorithms.regression.gp_regressors import *
from neu_ml.code.utils.compute_kernel_func_vals import \
    compute_kernel_func_vals

def plot_latent_trajectories(inputs, targets, kernel, phi_instances, \
                             phi_features, feature, precision, \
                             pred_ids = None, prob_thresh = 0.01):
    """Plot the latent trajectories as discovered by the dual_ibp_gp_regression
    algorithm.

    This plotter is a companion to the 'dual_ibp_gp_regression' algorithm and
    takes as input the 'phi_instances' and 'phi_featuers' arrays that the
    algorithm produces. The plotter will plot all latent trajectories for
    the specified observed feature provided the trajectories are more than
    negligibly probable (set by a threshold).

    Parameters
    ----------
    inputs : array, shape ( N, Q )
        Array of data inputs where 'N' is the number of instances and 'Q' is
        the dimension of the input space.

    targets : array, shape ( N, D )
        Array of data target values corresponding to the 'inputs'. 'N' is the
        number of instances, and 'D' is the dimension of the target space.

    kernel : list
        A list with two elements. The first element is a string indicating the
        type of kernel. The second element is an array indicating the parameter
        values of the kernel.

    phi_instances : array, shape ( N, K )
        Corresponds to the parameters of the binary matrix distribution
        describing the association between instances and latent features.
        'N' is the number of input points, and 'K' is the number of
        latent features used in the approximated beta process. 

    phi_features : array, shape ( D, K )
        Corresponds to the parameters of the binary matrix distribution
        describing the association between observed features and latent
        features. 'D' is the number of observed features, and 'K' is the number
        of latent features used in the approximated beta process.

    feature : int
        The target dimension for which to plot the trajectories. The inputs
        and corresponding targets will be rendered with scatter plots, and the
        trajectories will be rendered as lines.

    precision : double
        The known noise precision on the target values of interest

    pred_ids : array, shape ( M ), optional
        The indices of the inputs to use when plotting (all inputs are used to
        construct the GP Gram matrix). Not necessary if the input dimension is
        only 1D. If the input dimension is 2D, a 3D scatter plot with surfaces
        representing the trajectories will be created. If the input dimension
        is greater than 2D, 'pred_ids' must be specified.

    prob_thresh : double, optional
        Latent trajectories will only be plotted provided that at least two
        instances have a probability of belonging to the trajectory that is
        greater than this specified amount.

    See also
    --------
    dual_ibp_gp_regression
    """
    assert feature >=0 and feature < targets.shape[1], \
        "Specified feature not supported by targets"
    
    assert inputs.shape[0] == targets.shape[0] == phi_instances.shape[0], \
        "Mismatch in the number of instances"

    assert targets.shape[1] == phi_features.shape[0], \
        "Mismatch in the number of latent features"

    if len(inputs.shape) == 1 or (pred_ids is not None and \
      pred_ids.shape[0] == 1):
        plot_1d_domain_(inputs, targets, kernel, phi_instances, phi_features, \
                        feature, precision, pred_ids, prob_thresh)

    if pred_ids is not None:
      if pred_ids.shape[0] == 1:
          plot_1d_domain_(inputs, targets, kernel, phi_instances,
                    phi_features, feature, precision, pred_ids, prob_thresh)

    if len(inputs.shape) == 2:
        if inputs.shape[1] == 2:
            plot_2d_domain_(inputs, targets, kernel, phi_instances, \
                    phi_features, feature, precision, pred_ids, prob_thresh)

def ep_mat_(phi_instances, phi_features):
    """Compute the matrix that indicates for each element the probability
    that the element was not generated from noise (i.e. that the element IS
    associated with some latent feature).

    Parameters
    ----------
    phi_instances : array, shape ( N, K ),
        Matrix of Bernoulli parameter values relating to the association
        between instances and latent features.    

    phi_features : array, shape ( D, K )
        Matrix of Bernoulli parameter values relating to the association
        between observed features and latent features.

    Returns
    -------
    ep_mat : array, shape ( N, D )
        'N' represents the instances and 'D' represents the observed features.
	Each element of the matrix is in the interval [0, 1] and indicates
	the probability that the corresponding instance is NOT associated with
	noise with respect to the corresponding features.
    """
    N = phi_instances.shape[0]
    D = phi_features.shape[0]

    ep_mat = np.zeros([N, D])
    for n in xrange(0, N):
	for d in xrange(0, D):
	    ep_mat[n, d] = 1 - \
		np.prod(1 - phi_instances[n, :]*phi_features[d, :])

    return ep_mat

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

def plot_1d_domain_(inputs, targets, kernel, phi_instances, phi_features, \
                    feature, precision, pred_ids = None, prob_thresh = 0.01):
    """Handles plotting for the case when the input domain is 1D.
    
    Parameters
    ----------
    inputs : array, shape ( N, Q )
        Array of data inputs where 'N' is the number of instances and 'Q' is
        the dimension of the input space.

    targets : array, shape ( N, D )
        Array of data target values corresponding to the 'inputs'. 'N' is the
        number of instances, and 'D' is the dimension of the target space.

    kernel : list
        A list with two elements. The first element is a string indicating the
        type of kernel. The second element is an array indicating the parameter
        values of the kernel.

    phi_instances : array, shape ( N, K )
        Corresponds to the parameters of the binary matrix distribution
        describing the association between instances and latent features.
        'N' is the number of input points, and 'K' is the number of
        latent features used in the approximated beta process. 

    phi_features : array, shape ( D, K )
        Corresponds to the parameters of the binary matrix distribution
        describing the association between observed features and latent
        features. 'D' is the number of observed features, and 'K' is the number
        of latent features used in the approximated beta process.

    feature : int
        The target dimension for which to plot the trajectories. The inputs
        and corresponding targets will be rendered with scatter plots, and the
        trajectories will be rendered as lines.

    precision : double
        The known noise precision on the target values of interest

    pred_ids : array, shape ( M ), optional
        The indices of the inputs to use when plotting (all inputs are used to
        construct the GP Gram matrix). Not necessary if the input dimension is
        only 1D. If the input dimension is 2D, a 3D scatter plot with surfaces
        representing the trajectories will be created. If the input dimension
        is greater than 2D, 'pred_ids' must be specified.

    prob_thresh : double, optional
        Latent trajectories will only be plotted provided that at least two
        instances have a probability of belonging to the trajectory that is
        greater than this specified amount.
    """
    plt.scatter(inputs, targets[:, feature], s = 200, facecolors = 'None', \
                edgecolors = 'k')

    K = phi_instances.shape[1]
    colors = get_colors(K)

    ep_mat = ep_mat_(phi_instances, phi_features)
    
    for k in xrange(0, K):
        prob_mat = np.diag(ep_mat[:, feature]*phi_instances[:, k]*\
                           phi_features[feature, k])

        # Only plot provided that at least two instances have a reasonable
        # probability of belonging to this latent trajectory
        if np.sum(np.diag(prob_mat) > prob_thresh) > 1:
            regressor = GPRegressor(kernel, inputs, targets[:, feature], \
                                    precision = precision, prob_mat = prob_mat)
            preds, vars = regressor.get_target_predictions_at_inputs()
            plt.plot(inputs, preds, color = colors[k], linewidth = 4)

    plt.show()

