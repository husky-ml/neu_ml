import pdb

import copy
from scipy.special import psi, gamma, gammaln
from numpy import trace, log, pi, sqrt
from scipy.linalg import inv, det, pinv2
import numpy as np
import networkx as nx
from neu_ml.code.algorithms.regression.gp_regressors import GPRegressor
from neu_ml.code.data.data_collector import DataCollector

def nonparametric_gp_regression(inputs, targets, kernel, beta, alpha, iters, K,
                                prob_thresh, mix_factor_thresh=0.1, repeats=1,
				constraints=None, collector=None,
				z_disc=None):
    """Performs non-parametric Gaussian Process (GP) regression. Multiple
    curves can potentially be fit to the input data; the number of curves
    is determined automatically.

    Parameters
    ----------
    inputs : array, shape ( n_instances, input_dim )
        Input data. 'n_instances' is the number of instances / data points.
        'input_dim' is the dimension of the input space

    targets : array, shape ( n_instances, target_dim )
        Targets corresponding to input data. 'n_instances' is the number of
        instances. 'target_dim' is the dimension of the target variables.
        An NxD matrix of targets corresponding to the inputs in the first
        cell array.'D' is the dimension of the target variables.)

    kernel : list
        A list with two elements. The first element is a string indicating the
        type of kernel. The second element is an array indicating the
        parameter values of the kernel.

    beta : float
        The (assumed known) precision describing the noise on the target values

    alpha : float
        Hyper parameter of the Beta destribution describing latent variable
        'v'. See below and 'NonparametricGaussianProcessRegression' for more
        detailed description.

    iters : int
        The number of iterations of the optimization to compute.

    K : int
        An integer indicating the number of elements in the truncated Dirichlet
        process.

    mix_factor_thresh : float, optional
        A value in the interval [0.0, 1.0] indicating the minimum mixing factor
	that a Gaussian Process needs to have in order to be included in the
	output. The higher this threshold value is, the more stringent the
	requirement for a Gaussian Process to be returned.

    repeats : integer, optional
        The number of times the algorithm is run with a random initializatin.
	After each run, the variational lower bound is computed, and a record
	is kept of the best performing settings for the unobserved variables;
	these are the ones returned by this function.

    prob_thresh : float
        The probability threshold is a scalar value in the interval (0, 1). It
        indicates the minimum value of component 'k' of the latent variable,
        'z', that a given point needs to have to be considered a possible
        member of regression curve 'k'. A value of 0.001 has been tested and
        works effectively. Setting the value too low runs the risk of numerical
        instability in the algorithm. Setting the value too high prevents
        points that may have a significant probability of belonging to a curve
        from "saying so".
    
    constraints : networkx graph, optional
        The constraints are encoded in a networkx graph, each node should
        indicate an instance index, and edges indicate constraints. Each
        edge should have a string attribute called 'constraint', which must
        either take a value of 'must_link' or 'cannot_link'. If constraints
        are specified, the variation inference update equations will take
        them into account. See
        'ConstrainedNonparametricGaussianProcessRegression' in the repositrory
        'Documentation' folder for a description of how the updates are
        incorporated.

    collector : Instance of DataCollector, optional
        'collector' is used to gather data during the optimization for later
        evaluation and analysis.

    z_disc : array, shape ( n_instances, K ), optional
        Corresponds to the parameters of the discrete distribution describing
        latent variable z. The nth row corresponds to the nth latent variable.
        'n_instances' is the number of input points, and 'K' is the number of
        regression functions in the truncated Dirichlet Proces. If specified,
        the algorithm will be initialized with this matrix. Otherwise, the
        matrix will be randomly generated. Each row must sum to 1, and each
        element must be in the interval [0, 1].

    Returns
    -------
    regressors : list of 'GPRegressor'
        The length of the list indicates the number of Guassian Process
        regression curves that have input-targets that are likely to belong to
        the process

    z_disc : array, shape ( n_instances, K )
        Corresponds to the parameters of the discrete distribution describing
        latent variable z. The nth row corresponds to the nth latent variable.
        'n_instances' is the number of input points, and 'K' is the number of
        regression functions in the truncated Dirichlet Proces.

    lower_bound : float
        The highest variational lower bound found after running inference for
        each of the requested 'repeats'.

    Notes
    -----
    There are three variables in our variational distribution: 'z','F', and
    'v'. We will indicate these throughout as 'z', 'f', and 'v', respectively.
    Also, we will truncate the number of states to an integer 'K'. 

    Latent variable 'z' is governed by a discrete distribution. For each sample
    and each state there is an associated parameter 'r'. We will collectively
    represent these parameters by an NxK matrix, 'z_disc'.

    Latent variable 'v' is governed by a Beta distribution. We will designate
    the first parameter of this distribution as 'v_beta_a' and the second
    parameter as 'v_beta_b' (the prefix 'v' is to indicate the latent variable
    'v'). 

    Latent variable 'F' is governed by a Gaussian distribution. These are the
    Gaussian Process regression curves. The key variable used is 'f_gp_cov', a
    collection of 'K' matrices.

    References
    ----------
    1. J. C. Ross, J. G. Dy, 'Nonparametric Gaussian Processes with
    Constraints', ICML, 2013

    See also
    --------
    compute_kernel_func_vals : For supported kernel types.

    GPRegressor : For more information about
        the 'regressors' list returned from this function.
    """
    if repeats < 1:
	raise ValueError("Number of repeats must be greater than 0")

    # The following two containers will be used to keep track of the "optimal"
    # results (those that correspond to the largest variational lower bound,
    # measured at the end of each of the repeats specified by the user)
    regressors_opt = None
    z_disc_opt = None    
    lower_bound_opt = np.finfo('d').min

    # If constraints have been specified, get the connected subgraphs
    constraint_subgraphs = []
    if constraints is not None:
        constraint_subgraphs = get_constraint_subgraphs(constraints)

    # Get the number of data points
    n_instances = inputs.shape[0]

    # Simply a list of indices that can be used to indicate which observations
    # are "active" for a given Gaussian Process (i.e. which observations are
    # likely to belong to the Gaussian Process). These are the indices of our
    # observations
    obs_indices = np.arange(0, n_instances)

    # Get the dimension of the target variables
    target_dim = targets.shape[1]

    for r in xrange(0, repeats):
        # Initialize the parameters for the beta distribution. 
        v_beta_a = np.zeros(K)
        v_beta_b = np.zeros(K)
	    
	if z_disc is None:
	    # Initialize 'z_disc' by creating random entries with the
	    # constraint that each row sums to 1
	    z_disc = np.random.rand(n_instances, K)

    	    for n in np.arange(0, n_instances):
	        z_disc[n, :] = z_disc[n, :]/sum(z_disc[n,:]);

        # Create a regressor. This regressor will be copied 'K' times to add
	# to the 'regressors' list. The difference between the regressors will
	# be the contained "active observations" index set and the contained
	# "probability matrix" indicating the probability of each input-target
	# pair belonging to the Gaussian Process
        tmp_regressor = GPRegressor(kernel, precision=beta, inputs=inputs,
                                    targets=targets)
        gram_mat = tmp_regressor.get_gram_mat()

        regressors = []
        for i in np.arange(0, K):
	    v_beta_a[i] = 1
            v_beta_b[i] = alpha

            regressors.append(copy.deepcopy(tmp_regressor))

        # Record the cluster assignments if a data collector is specified
        if collector is not None:
	    update_data_collector(z_disc, collector)

        # Now iteratively update the parameters of the various variational
        # distributions
        for i in np.arange(0, iters):
            # Given 'z_disc' (i.e. the probability of each data point belonging
	    # to a given regression curve), we can update the various Gaussian
            # Processes. We do this by computing the predicted means and
	    # variances at each input location for all the Gaussian Processes,
	    # but each process is described by the the input-target pairs
	    # "likely" to belong to that process. Note that this step is used
	    # (as opposed to using each Gaussian Process mean vector and
	    # covariance matrix directly) because the mean vectors and
	    # covariance matrices of each GP in general change in size from
	    # iteration to iteration in response to how many points likely
	    # belong to the regression curve. Theoretically, we need not make
	    # this allowance (we could maintain the same size for all GPs from
	    # iteration to iteration). However, numerically things begin to
	    # get quite tricky when we attempt to compute a GP covariance
	    # matrix when the targets/inputs have only miniscule probability
	    # of belonging to that GP. This is why we introduce the
	    # 'prob_thresh': to update a given GP covariance matrix, we only
	    # consider those inputs/targets that have some nominal probability
	    # (designated with the 'prob_thresh') of belonging to the curve.
	    means = []
	    vars  = []
	    for k in np.arange(0, K):
	        # By default, set 'means' and 'variances' to be huge values.
		# This is one way of representing the "impossible curve": we
		# need place-holder values for curves for which none of our
		# inputs / targets have any reasonable probability of
		# membership.
	        means.append(np.finfo('f').max*np.ones((n_instances,
					target_dim), dtype=float))
	        vars.append( np.finfo('f').max*np.ones((n_instances,
						target_dim), dtype=float))

	        if np.sum(np.nonzero(z_disc[:,k]>=prob_thresh)) > 0:            
                    # Note that here we're computing 'means' and 'variances'
                    # corresponding to every input, but we're doing it with GP
                    # covariance matrices that are potentially smaller than
                    # NxN, where 'N' here is the number of inputs. This is
                    # accomplished by using the "active indices" specification
                    # mechanism provided by the classes in 'gp_regressors'. At
                    # each iteration, we designate the active index set to be
                    # those indices corresponding to input-target pairs likely
                    # to belong to the Gaussian Process under considerations.

                    # Construct the probabilty matrix
                    prob_mat = np.diag(z_disc[:,k])

                    # Now get the currently active observations, determined by
                    # input-target pairs having "significant" probability of
                    # belonging to the current Gaussian Process (as determined
                    # by probability being greater than 'prob_thresh')
                    active_obs = np.delete(obs_indices,
                                           np.nonzero(z_disc[:,k] <
                                                      prob_thresh), axis=0)
                    regressors[k].set_active_obs_indices(active_obs)
                    regressors[k].set_target_probabilities(prob_mat)

                    [means[k], vars[k]] = \
                        regressors[k].get_target_predictions_at_inputs()
        
            # Now given the 'means' and 'vars' computed above for each input,
            # update the probability of membership of each data point to each
	    # curve
            z_disc, z_disc_r = update_Z_params(targets, beta, v_beta_a,
				v_beta_b, K, means, vars,
				constraint_subgraphs, gram_mat, prob_thresh) 

            # Record the cluster assignments if a data collector is specified
            if collector is not None:
                update_data_collector(z_disc, collector)
 
	    print np.round(z_disc.sum(axis=0))

	    # Update parameters for the random variable 'v'
	    [v_beta_a, v_beta_b] = update_V_params(z_disc, K, alpha)    

	# Compute the variational lower bound. It should decrease at every
	# iteration
	bound = lower_bound(targets, gram_mat, regressors, z_disc, z_disc_r,
			    beta, alpha, constraint_subgraphs, prob_thresh)

	if bound > lower_bound_opt:
	    lower_bound_opt = bound

	    del z_disc_opt
	    del regressors_opt
		
	    z_disc_opt = z_disc.copy()
	    regressors_opt = list(regressors)

 	del z_disc
	z_disc = None
	del regressors

    # Before returning, remove from regressors any Gaussian Processes that
    # don't have any input-targets likely assigned to them
    keep_looking = True
    while keep_looking == True:
        keep_looking = False
	for k in np.arange(0, len(regressors_opt)):
	    if np.sum(regressors_opt[k].get_prob_mat())/n_instances < \
                    mix_factor_thresh:
		del regressors_opt[k]
		keep_looking = True
		break
 
    return regressors_opt, z_disc_opt, lower_bound_opt

def update_Z_params(targets, beta, v_beta_a, v_beta_b, K, means, vars,
                    constraint_subgraphs, gram_mat, prob_thresh):
    """This function updates the matrix 'R', which indicates the probability
    of each point belonging to each of 'K' possible cluster groups.

    Parameters
    ----------
    targets : array, shape ( n_instances, input_dim )
        Matrix of target values. 'n_instances' is the number of instances /
        data points. 'input_dim' is the dimension of the input space

    beta : float
        The (known) precision of the Gaussian noise on the targets

    v_beta_a : array, shape ( K )
        Array of scalars. The kth entry is the first parameter of the Beta
        distribution describing the kth latent variable, 'v'. (See companion
        documentation 'NonparametricGaussianProcessRegression.pdf' for more
        description of latent variables)

    v_beta_b : array, shape ( K )
        Array of scalars. The kth entry is the second parameter of the Beta
        distribution describing the kth latent variable, 'v'. (See companion
        documentation 'NonparametricGaussianProcessRegression.pdf' for more
        description of latent variables

    K : int
        An integer indicating the number of elements in the truncated Dirichlet
        process

    means : list
        This is a K-element list. The kth element is a (n_instances)x(target_dim)
        array, where 'n_instances' is the number of samples, and 'target_dim'
        is the dimension of the target variable. The (n,d) entry of element 'k'
        is the mean value of Gaussian Process 'k' at the nth input value for
        target dimension 'd'.

    vars : list
        This is a K-element list. The kth entry is a (n_instances)x(target_dim)
        array, where 'n_instances' is the number of samples, and 'target_dim'
        is the dimension of the target variable. The (n,d) entry of cell 'k'
        is the variance of Gaussian Process 'k' at the nth input value for
        target dimension 'd'.

    constraint_subgraphs : list of networkx graphs
        Each element in the list is a connected sub-graph of 'constraints'.
        Each node in the graph represents a random variable, and each edge
        represents a constraint. The edges must have a string attribute called
        'constraint' which can be set to either 'must_link' or 'cannot_link'

    gram_mat : array, shape ( n_instances, n_instances )
        The Gram matrix computed over the inputs. Used for weighting the effect
        of each of the constraints. Constraints corresponding to instances that
        have a high kernel value will be weighted more heavily than those that
        have a lower kernel value.
	
    prob_thresh : float
        The probability threshold is a scalar value in the interval (0, 1). It
        indicates the minimum value of component 'k' of the latent variable,
        'z', that a given point needs to have to be considered a possible
        member of regression curve 'k'. Any state configuration having an
        individual instance with a probability less than this threshold will be
        considered an impossible configuration. This greatly increase
        computational efficiency

    Returns
    -------
    z_disc : array, shape ( n_instances, K )
        Array corresponding to the parameters of the discrete distribution
        describing latent variable z. The nth row corresponds to the nth latent
        variable, z_n.

    z_disc_r : array, shape ( n_instacnes, K )
        Array corresponding to the 'r' term used to compute parameters of the
	discrete distribution describing latent variable z. The nth row
	corresponds to the nth latent variable, z_n. Note that if no
	constraints are used, this array will be identical to 'z_disc'.

    Details
    -------    
    In order to compute 'z_disc' we must take the ln of Eq. 17 in 
    'NonparametricGaussianProcessRegression.pdf'. However, directly computing
    the ln can lead to numerical issues. Instead, we decompose the right hand
    side into base 10 mantissas and exponents. Subsequent processing is then
    stable. 
    """
    
    # Get the number of data points and the dimesnion of the target variables
    n_instances = targets.shape[0]
    target_dim  = targets.shape[1]
    
    # The following equalities are standard for the Beta distribution (second
    # two). These are the expectations that appear in Eq. 17 of
    # 'NonparametricGaussianProcessRegression'
    expect_ln_v = np.zeros(K)
    expect_ln_1_minus_v = np.zeros(K)
    for i in np.arange(0, K):
        expect_ln_v[i] = psi(v_beta_a[i]) - \
                         psi(v_beta_a[i] + v_beta_b[i]);
        expect_ln_1_minus_v[i] = psi(v_beta_b[i]) - \
                                 psi(v_beta_a[i] + v_beta_b[i]);
    
    # Compute the 'Rho' matrix. Each entry of the matrix corresponds to taking
    # the exponential of the Eq. 17 in 'NonparametricGaussianProcessRegression'
    rho = np.zeros((n_instances, K))
    mantissas = np.zeros((n_instances, K))
    exponents_base_10 = np.zeros((n_instances, K))
    
    const = np.log(np.sqrt(beta/(2.0*np.pi)))
    for n in np.arange(0, n_instances):
        for k in np.arange(0, K):
	    tmp_sum = expect_ln_v[k]
	    for i in np.arange(0, k):
	        tmp_sum = tmp_sum + expect_ln_1_minus_v[i]
		
            # See Eq. 17 in 'NonparametricGaussianProcessRegression' for the
            # following expression. Decompose the right hand side of Eq. 17
            # into base 10 exponents and mantissas
            exponent_base_e = 0.0
	    for d in np.arange(0, target_dim):
	        exponent_base_e = exponent_base_e + const - \
                    (beta/2.0)*((targets[n,d] - means[k][n,d])**2.0 + \
				vars[k][n,d])

            exponent_base_e = exponent_base_e + tmp_sum
	    
            exponents_base_10[n, k] = np.floor(exponent_base_e/np.log(10.0))
            mantissas[n, k] = 10.0**((exponent_base_e/np.log(10.0)) - \
		exponents_base_10[n, k]);
    
    # What we really care about is the normalization of each row of 'rho'. That
    # being the case, we can subtract from each row of 'exponents_base_10' the
    # maximum value in that row.
    exponents_base_10_shift = np.zeros((n_instances, K))
    for n in np.arange(0, n_instances):
        exponents_base_10_shift[n,:] = exponents_base_10[n,:] - \
	    np.max(exponents_base_10[n, :])

    # Now compute the 'rho' matrix. The elements correspond to rho_{n,k} seen in
    # Eq. 17
    for n in np.arange(0, n_instances):
        rho[n, :] = mantissas[n, :]*10.0**(exponents_base_10_shift[n, :]);
    
    # See Eq. 19. Need to normalize to get the matrix holding the elements,
    # r_{n,k} (i.e. the 'z_disc' matrix)
    z_disc = np.zeros((n_instances, K))
    for n in np.arange(0, n_instances):
        z_disc[n, :] = rho[n,:]/np.sum(rho[n, :])

    z_disc_r = z_disc.copy()

    # Now apply constraints
    num_subgraphs = len(constraint_subgraphs)

    for i in xrange(0, num_subgraphs):
        update_Z_params_rows(z_disc, gram_mat, constraint_subgraphs[i],
			     prob_thresh)

    return z_disc, z_disc_r

def update_V_params(z_disc, K, alpha):
    """Computes the first parameter of the Beta distributions for latent
    variable 'v' in the variational approximation.

    Parameters
    ----------
    z_disc : array, shape ( n_instances, K )
        Corresponds to the parameters of the discrete distribution describing
        latent variable z. The nth row corresponds to the nth latent variable.
        'n_instances' is the number of input points, and 'K' is the number of
        regression functions in the truncated Dirichlet Proces. See
        'NonparametricGaussianProcessRegression' and in 'Documentation' for
        more complete descriptions of latent variables and their naming in this
        implementation.

    K : int
        An integer indicating the number of elements in the truncated Dirichlet
        process.

    alpha : float
        Hyper parameter of the Beta destribution describing latent variable
        'v'. See below and 'NonparametricGaussianProcessRegression' in
        'Documentation' for more detailed descriptions.

    Returns
    -------
    v_beta_a : array, shape ( K )
        Array of scalars. The kth entry is the first parameter of the Beta
        distribution describing the kth latent variable, 'v'. (See companion
        documentation 'NonparametricGaussianProcessRegression.pdf' for more
        description of latent variables). 'K' is the number of components in
        the Dirichlet Process (truncated)

    v_beta_b : array, shape ( K )
        Array of scalars. The kth entry is the second parameter of the Beta
        distribution describing the kth latent variable, 'v'. (See companion
        documentation 'NonparametricGaussianProcessRegression.pdf' for more
        description of latent variables. 'K' is the number of components in
        the Dirichlet Process (truncated)

    Details
    -------
    See Eq. 43 in 'NonparametricGaussianProcessRegression'    
    """

    v_beta_a = np.zeros(K)
    v_beta_b = np.zeros(K)

    for k in np.arange(0, K):
        v_beta_a[k] = 1.0 + np.sum(z_disc[:,k]) 

        tmpSum = 0.0
        for i in np.arange(k+1, K):
            tmpSum = tmpSum + np.sum(z_disc[:, i])

        v_beta_b[k] = alpha + tmpSum

    return v_beta_a, v_beta_b

def get_constraint_subgraphs(constraints):
    """Pulls out the connected subgraphs within the 'constraints' graph and
    puts them into a list for easier access.

    Parameters
    ----------
    constraints : networkx graph
        The constraints are encoded in a networkx graph, each node should
        indicate an instance index, and edges indicate constraints. Each
        edge should have a string attribute called 'constraint', which must
        either take a value of 'must_link' or 'cannot_link'. Generally, the
        graph describing 'constraints' will be composed of connected subgraphs.
        This routine isolates the connected subgraphs.

    Returns
    -------
    constraint_subgraphs : list of networkx graphs
        Each element in the list is a connected sub-graph of 'constraints'
    """
    num_subgraphs = len(nx.connected_components(constraints))

    constraint_subgraphs = []
    for i in np.arange(0, num_subgraphs):
        tmp_graph = nx.subgraph(constraints, nx.connected_components(constraints)[i])
        constraint_subgraphs.append(tmp_graph)

    return constraint_subgraphs

def update_Z_params_rows(z_disc, gram_mat, constraint_subgraph, prob_thresh):
    """Updates 'z_disc' according to the constraints supplied in
    'constraint_subgraph'

    The nodes in the subgraph designate which rows (variables) to update in
    'z_disc', the matrix that encodes the expected values of each indicator
    variable, Z. See 'ConstraintedNonparametricGaussianProcessRegression' in
    the repository 'Documentation' folder for details on the update equations.
    This function does not return a value but rather updates the proper rows
    of 'z_disc'.

    Parameters
    ----------
    z_disc : array, shape ( n_instances, K )
        Array corresponding to the parameters of the discrete distribution
        describing latent variable z. The nth row corresponds to the nth latent
        variable, z_n.

    gram_mat : array, shape ( n_instances, n_instances )
        The Gram matrix computed over the inputs. Used for weighting the effect
        of each of the constraints. Constraints corresponding to instances that
        have a high kernel value will be weighted more heavily than those that
        have a lower kernel value.

    constraint_subgraph : networkx graph
        The constraints are encoded in a networkx graph, each node should
        indicate an instance index, and edges indicate constraints. Each
        edge should have a string attribute called 'constraint', which must
        either take a value of 'must_link' or 'cannot_link'.

    prob_thresh : float
        The probability threshold is a scalar value in the interval (0, 1). It
        indicates the minimum value of component 'k' of the latent variable,
        'z', that a given point needs to have to be considered a possible
        member of regression curve 'k'. Any state configuration having an
        individual instance with a probability less than this threshold will be
        considered an impossible configuration. This greatly increase
        computational efficiency
    """
    # Collect information about the instance IDs represented by the nodes in
    # the 'constraint_subgraph'. Note that the 'nodes()' operation on the
    # graph produces a sorted list by default. We sort again for security,
    # esp. given that we don't expect a large number of nodes in any one
    # subgraph, so the sort operation introduces minimal overhead.
    num_nodes = len(constraint_subgraph.nodes())	
    constraint_subgraph.nodes().sort()
    node_ids = constraint_subgraph.nodes()    

    # We will need to consider all possible 'state' configurations. Each of
    # variables can be in one of K states, for a total of K^(num_nodes)
    # possibilities. We will represent the state of the collection of variables
    # with the matrix 'state_mat'
    num_states = z_disc.shape[1]
    num_state_mats = num_states**num_nodes

    # Now loop over all possible state matrices. For each state matrix, compute
    # the unnormalized probability (this is the quantity in Eq. 20 in the
    # 'ConstraintedNonparametricGaussianProcessRegression' document, withouth
    # the partition function). Each computed probability is multiplied by the
    # corresponding state matrix and added to 'state_mat_accum'. The total
    # unnormalized probability is accumulated in 'prob_accum'. We take
    # advantage of the fact that if a given instance has a probability lower
    # than 'prob_thresh' for a given state, we can effectively consider
    # any configuration including that state to be "impossible". This greatly
    # reduces the computational burden required: for each instance in the
    # subgraph, we need only record the columns in 'z_disc' for which
    # its probability is >= 'prob_thresh'. We do this for each of the instances
    # in the subgraph and only loop over those state configurations.
    sig_cols = []
    num_state_mats = 1.0

    for i in xrange(0, num_nodes):
	cols = (np.nonzero(z_disc[node_ids[i], :] > prob_thresh)[0]).tolist()
	sig_cols.append(cols)
	num_state_mats *= len(sig_cols[i])

    # Initialize the state matrix
    state_mat = np.zeros([num_nodes, num_states])	
    for n in xrange(0, num_nodes):
	state_mat[n, sig_cols[n][0]] = 1.0

    state_mat_accum = np.zeros([num_nodes, num_states])
    prob_accum = 0.0

    for i in np.arange(0, num_state_mats):        
        prob = compute_subgraph_unnormalized(constraint_subgraph, gram_mat,
                                             z_disc, state_mat)            
        state_mat_accum += prob*state_mat
        prob_accum += prob
        inc_state_mat(state_mat, sig_cols)

    # Now create the normalized matrix
    normalized_mat = (1.0/prob_accum)*state_mat_accum

    # Lastly, we have to insert the rows of 'normalized_mat' properly into
    # 'z_disc'. Each row of 'normalized_mat' corresponds to an instance, and
    # the rows are ordered so that the first row corresponds to the smallest
    # instance ID, the second row corresponds to the next smallest instance ID,
    # etc.     
    for i in np.arange(0, num_nodes):
        z_disc[node_ids[i], :] = normalized_mat[i, :]

def inc_state_mat(state_mat, sig_cols=None):
    """Increment 'state_mat' by one unit.

    This function takes as input a matrix, 'state_mat', in which each row has
    one element equal to 1.0 and every other element equal to 0.0. The
    algorithm proceeds by moving down the rows and moving the row element equal
    to 1.0 over by one place. If this means "resetting" that element to the
    first position, then the next row will be considered for the same procedure.
    This continues until no "resets" occur.

    Parameters
    ----------
    state_mat : array, shape ( num_rows, num_cols )
        A matrix in which each row has exactly one element equal to 1.0. The
        other elements are equal to 0.0

    sig_cols : list of 'num_rows' arrays shaped ( n_cols ), optional
        Each row can have a different number of entries. Each list entry
	indicates the set of columns (possible states) that the corresponding
	data instance can take on. If not specified, it's assumed that each
	row (instance) can take on every possible state. If specified, columns
	taking on the value 1.0 will be moved to the next allowable column
	(state)        
    """
    num_rows = state_mat.shape[0]
    num_cols = state_mat.shape[1]
	
    row_inc = 0
    reset = True
    while row_inc < num_rows and reset:
        # For the current row, find the column whose element is 1.0
        col = np.nonzero(state_mat[row_inc,:]==1.0)[0][0]

	if sig_cols is not None:
            num_sigs = len(sig_cols[row_inc])
	    index = sig_cols[row_inc].index(col)
            if num_sigs - 1 == index:
	        reset = True
	        state_mat[row_inc, col] = 0.0
	        state_mat[row_inc, sig_cols[row_inc][0]] = 1.0
	        row_inc += 1
	    else:
	        reset = False
		state_mat[row_inc, col] = 0.0
		state_mat[row_inc, sig_cols[row_inc][index+1]] = 1.0
        else:
            if col == num_cols-1:
	        reset = True
	        state_mat[row_inc, col] = 0.0
	        state_mat[row_inc, 0] = 1.0
	        row_inc += 1
            else:
	        reset = False
	        state_mat[row_inc, col] = 0.0
	        state_mat[row_inc, col+1] = 1.0

def compute_subgraph_unnormalized(constraint_subgraph, gram_mat, z_disc,
                                  state_mat):
    """Computes the quantity inside the brackets in Eq. 20 in
    'ConstrainedNonparametricGaussianProcessRegression' (without the partition
    function), for a given state matrix.

    Parameters
    ----------
    constraint_subgraph : networkx graph
        The constraints are encoded in a networkx graph, each node should
        indicate an instance index, and edges indicate constraints. Each
        edge should have a string attribute called 'constraint', which must
        either take a value of 'must_link' or 'cannot_link'. 

    gram_mat : array, shape ( n_instances, n_instances )
        The Gram matrix computed over the inputs. Used for weighting the effect
        of each of the constraints. Constraints corresponding to instances that
        have a high kernel value will be weighted more heavily than those that
        have a lower kernel value.

    z_disc : array, shape ( n_instances, K )
        Array corresponding to the parameters of the discrete distribution
        describing latent variable z. The nth row corresponds to the nth latent
        variable, z_n.

    state_mat : array, shape ( num_rows, num_cols )
        A matrix in which each row has exactly one element equal to 1.0. The
        other elements are equal to 0.0. There should be the same number of
        rows as there are nodes in 'constraint_subgraph'

    Returns
    -------
    unnormalized_prob : float
        The quantity inside the brackets in Eq. 20 in
        'ConstrainedNonparametricGaussianProcessRegression', without the
        partition function.
    """

    num_nodes = len(constraint_subgraph.nodes())
    constraint_subgraph.nodes().sort()
    node_ids = constraint_subgraph.nodes()
    edges = constraint_subgraph.edges()

    unnormalized_term = 1.0
    for i in np.arange(0, num_nodes):
        id1 = node_ids[i]
        row1 = node_ids.index(id1)
        state_mat_row1 = state_mat[row1, :]
        energy_accum = 0.0
        for j in np.arange(0, num_nodes):
            id2 = node_ids[j]
            row2 = node_ids.index(id2)
            state_mat_row2 = state_mat[row2, :]
            if (id1, id2) in edges:
		try:
                    kernel_val = gram_mat[id1, id2]
		except IndexError:
		    print "IndexError"
		    pdb.set_trace()
                energy = compute_energy(state_mat_row1, state_mat_row2,
                            constraint_subgraph[id1][id2]['constraint'],
                            kernel_val)
                energy_accum += energy
            elif (id2, id1) in edges:
                kernel_val = gram_mat[id2, id1]
                energy = compute_energy(state_mat_row1, state_mat_row2,
                            constraint_subgraph[id2][id1]['constraint'],
                            kernel_val)
                energy_accum += energy

            unnormalized_term *= np.exp(-energy_accum)

    # Now we need to compute the contribution of the r_(n,k) terms show in
    # Eq. 20
    r_term = 1.0
    for i in np.arange(0, num_nodes):
        col = np.nonzero(state_mat[i, :] == 1.0)[0][0]
        r_term *= z_disc[node_ids[i], col]

    unnormalized_term *= r_term

    return unnormalized_term

def get_weight():
    """Get the weight used for computing the MRF energy term.

    Returns
    -------
    weight : float
        The weight used for computing the MRF energy term.
    """
    # Set a weight for energy function. Adjusting this value will give different
    # behaviour: the larger the weight, the more influence the constraints have.
    # The weight value should be >= 0.0	
    return 10.0	

def compute_energy(state_mat_row1, state_mat_row2, constraint, kernel_val):
    """Computes the energy function value represented by 'H' in Eq. 20 of
    'ConstrainedNonparametricGaussianProcessRegression'

    Parameters
    ----------
    state_mat_row1 : array, shape( 1, K )
        A selected row from the state matrix. This row indicates the state (ie.
        which Gaussian Process regression curve this instance belongs to). 'K'
        is the number of elements in the truncated Dirichlet process.

    state_mat_row2 : array, shape( 1, K )
        A selected row from the state matrix. This row indicates the state (ie.
        which Gaussian Process regression curve this instance belongs to). 'K'
        is the number of elements in the truncated Dirichlet process.

    constraints : string
        Takes on 'must_link' or 'cannot_link' to indicate the constraint type
        between the two instances corresponding to 'state_mat_row1' and
        'state_mat_row2'

    kernel_val : float
        This is the Gram matrix entry corresponding to the instances
        represented by 'state_mat_row1' and 'state_mat_row2'. It is used to
        weight the effect of the given constraint. The higher the kernel
        value, the more weight the constraint will have

    Returns
    -------
    energy : float
        The computed energy term.
    """

    # Compute the inner product of the two rows. If the inner product is 1.0,
    # then the two instances are in the same state, otherwise they are not.
    inner_product = np.dot(state_mat_row1, state_mat_row2)

    # Get the weight for computation of the energy function
    weight = get_weight()

    energy = 0.0
    if inner_product == 1.0:
        if constraint == 'must_link':
            energy = -weight
    else:
        if constraint == 'cannot_link':
            energy = -weight

    return energy

def compute_energy_expect(z_disc, constraint_subgraph):
    """Compute the expectation of the MRF energy term over a given subgraph

    Parameters
    ----------
    z_disc : array, shape ( n_instances, K )
        Array corresponding to the parameters of the variational discrete
	distribution describing latent variable z. The nth row corresponds to
	the nth latent variable, z_n.

    constraint_subgraph : networkx graph
        The constraints are encoded in a networkx graph, each node should
        indicate an instance index, and edges indicate constraints. Each
        edge should have a string attribute called 'constraint', which must
        either take a value of 'must_link' or 'cannot_link'.

    Returns
    -------
    expect_energy : float
        The expectation of the MRF energy term computed with respect to the
	specified subgraph. The expectation is taken with respect to the
        variational distribution for latent variable Z.
    """
    expect_energy = 0.0

    # Get the weight used for computating the energy term
    weight = get_weight()
   	
    # We will consider every edge (i.e. constraint) in the subgraph. For a
    # given constraint, we will take the element-wise product (inner product)
    # of the two corresponding rows in z_disc. This quantity is proportional
    # to the probability that the two latent variables are in the same state.
    # We will call this variable 'prop_same'. We will then take the inner
    # product of z_disc[i,:] and (1-z_disc[j,:]) and add this to the inner
    # product of (1-z_disc[i,:]) and z_disc[j,:], which is proportional to
    # the probability that z_i and z_j are in different states. This will be
    # called 'prop_diff'.
    edges = constraint_subgraph.edges()
    for i in xrange(0, len(edges)):
	id1 = edges[i][0]
	id2 = edges[i][1]
	z1_1 = z_disc[id1, :] #Prob that z1 is 1 in each state
	z2_1 = z_disc[id2, :] #Prob that z2 is 1 in each state
	z1_0 = 1. - z_disc[id1, :] #Prob that z1 is 0 in each state
	z2_0 = 1. - z_disc[id2, :] #Prob that z2 is 0 in each state	
	prop_same = np.dot(z1_1, z2_1)
	prop_diff = np.dot(z1_1, z2_0) + np.dot(z1_0, z2_1)

	# Now Pr(<z1, z2>=1) is:
	prob_same = prop_same/(prop_same + prop_diff)
	# and Pr(<z1, z2>=0) is:
	prob_diff = prop_diff/(prop_same + prop_diff)

	# Finally, update the cumulative expected energy for this subgraph
	if constraint_subgraph[id1][id2]['constraint'] == 'must_link':
	    expect_energy -= weight*prob_same
	if constraint_subgraph[id1][id2]['constraint'] == 'cannot_link':		
            expect_energy -= weight*prob_diff
	
    return expect_energy

def update_data_collector(z_disc, collector):
    """Update the data collector with the current cluster assignments.

    Parameters
    ----------
    z_disc : array, shape ( n_instances, K )
        Array corresponding to the parameters of the variational discrete
	distribution describing latent variable z. The nth row corresponds to
	the nth latent variable, z_n.

    collector : Instance of DataCollector
        'collector' is used to gather data during the optimization for later
        evaluation and analysis.
    """
    num_instances = z_disc.shape[0]
    assignments = np.zeros(num_instances)
    
    for i in np.arange(0, num_instances):
        max_val = np.max(z_disc[i,:])
        cluster_assignment = np.nonzero(z_disc[i,:] == max_val)[0][0]
        assignments[i] = cluster_assignment

    trial = collector.get_latest_cluster_assignment_trial()
    run = collector.get_latest_cluster_assignment_run(trial)
    collector.add_new_iteration(trial, run)
    iteration = collector.get_latest_cluster_assignment_iteration(trial, run)

    collector.set_cluster_assignments(assignments, trial, run, iteration)

def compute_partition_func(z_disc, gram_mat, constraint_subgraph,
                           prob_thresh):
    """Compute the partition function for a given constraint subgraph

    Parameters
    ----------
    z_disc : array, shape ( n_instances, K )
        Array corresponding to the parameters of the discrete distribution
        describing latent variable z. The nth row corresponds to the nth latent
        variable, z_n.

    gram_mat : array, shape ( n_instances, n_instances )
        The Gram matrix computed over the inputs. Used for weighting the effect
        of each of the constraints. Constraints corresponding to instances that
        have a high kernel value will be weighted more heavily than those that
        have a lower kernel value.

    constraint_subgraph : networkx graph
        The constraints are encoded in a networkx graph, each node should
        indicate an instance index, and edges indicate constraints. Each
        edge should have a string attribute called 'constraint', which must
        either take a value of 'must_link' or 'cannot_link'.

    prob_thresh : float
        The probability threshold is a scalar value in the interval (0, 1). It
        indicates the minimum value of component 'k' of the latent variable,
        'z', that a given point needs to have to be considered a possible
        member of regression curve 'k'. Any state configuration having an
        individual instance with a probability less than this threshold will be
        considered an impossible configuration. This greatly increase
        computational efficiency.
	
    Returns
    -------
    partition_func : float
        The partition function value needed to normalize the specified subgraph
    """
    # Collect information about the instance IDs represented by the nodes in
    # the 'constraint_subgraph'. Note that the 'nodes()' operation on the
    # graph produces a sorted list by default. We sort again for security,
    # esp. given that we don't expect a large number of nodes in any one
    # subgraph, so the sort operation introduces minimal overhead.
    num_nodes = len(constraint_subgraph.nodes())	
    constraint_subgraph.nodes().sort()
    node_ids = constraint_subgraph.nodes()    

    # We will need to consider all possible 'state' configurations. Each of
    # variables can be in one of K states, for a total of K^(num_nodes)
    # possibilities. We will represent the state of the collection of variables
    # with the matrix 'state_mat'
    num_states = z_disc.shape[1]

    # Now loop over all possible state matrices. For each state matrix, compute
    # the unnormalized probability (this is the quantity in Eq. 20 in the
    # 'ConstraintedNonparametricGaussianProcessRegression' document, without
    # the partition function). The total unnormalized probability is
    # to produce the value for the partition function. We take advantage of the
    # fact that if a given instance has a probability lower than 'prob_thresh'
    # for a given state, we can effectively consider any configuration
    # including that state to be "impossible". This greatly reduces the
    # computational burden required: for each instance in the subgraph, we need
    # only record the columns in 'z_disc' for which its probability is
    # >= 'prob_thresh'. We do this for each of the instances in the subgraph
    # and only loop over those state configurations.
    sig_cols = []
    num_state_mats = 1.0

    for i in xrange(0, num_nodes):
	cols = (np.nonzero(z_disc[node_ids[i], :] > prob_thresh)[0]).tolist()
	sig_cols.append(cols)
	num_state_mats *= len(sig_cols[i])

    # Initialize the state matrix
    state_mat = np.zeros([num_nodes, num_states])	
    for n in xrange(0, num_nodes):
	state_mat[n, sig_cols[n][0]] = 1.0

    partition_func = 0.0

    for i in np.arange(0, num_state_mats):        
        prob = compute_subgraph_unnormalized(constraint_subgraph, gram_mat,
                                             z_disc, state_mat)            
        partition_func += prob
        inc_state_mat(state_mat, sig_cols)    

    #print partition_func
    return partition_func

def lower_bound(targets, gram_mat, regressors, z_disc, z_disc_r, beta, alpha,
		constraint_subgraphs, prob_thresh):
    """Compute the variational lower bound

    Parameters
    ----------
    targets : array, shape ( n_instances, target_dim )
        Targets corresponding to input data. 'n_instances' is the number of
        instances. 'target_dim' is the dimension of the target variables.
        An NxD matrix of targets corresponding to the inputs in the first
        cell array.'D' is the dimension of the target variables.)

    gram_mat : array, shape ( n_instances, n_instances )
        The Gram matrix computed over the inputs. 

    regressors : list of 'GPRegressor'
        Gaussian Process regression curves.

    z_disc : array, shape ( n_instances, K )
        Corresponds to the parameters of the discrete distribution describing
        latent variable z. The nth row corresponds to the nth latent variable.
        'n_instances' is the number of input points, and 'K' is the number of
        regression functions in the truncated Dirichlet Proces.

    z_disc_r : array, shape ( n_instances, K )
        Array corresponding to the 'r' term used to compute parameters of the
	discrete distribution describing latent variable z. The nth row
	corresponds to the nth latent variable, z_n. Note that if no
	constraints are used, this array will be identical to 'z_disc'.

    beta : float
        The (assumed known) precision describing the noise on the target values

    alpha : float
        Hyper parameter of the Beta destribution describing latent variable
        'v'. See below and 'NonparametricGaussianProcessRegression' for more
        detailed description.

    constraint_subgraphs : list of networkx graphs
        Each element in the list is a connected sub-graph of 'constraints'.
        Each node in the graph represents a random variable, and each edge
        represents a constraint. The edges must have a string attribute called
        'constraint' which can be set to either 'must_link' or 'cannot_link'

    prob_thresh : float
        The probability threshold is a scalar value in the interval (0, 1). It
        indicates the minimum value of component 'k' of the latent variable,
        'z', that a given point needs to have to be considered a possible
        member of regression curve 'k'. 
        
    Returns
    -------
    lower_bound : float
        The variational lower bound
    """
    # First get some values that will be used throughout this function
    D = targets.shape[1] #Dimension of target variables
    N = targets.shape[0] #Number of instances
    K = len(regressors) #Number of regressors from truncated stick-breaking

    # There are seven terms that, summed together, comprise the lower bound.
    # We compute each term individually instead of grouping and re-arranging
    # in order keep the implementation clearer. Start with the first term:
    term1 = 0.0
    
    for k in xrange(0, K):
	# We won't update term1 if the current regressor under
	# consideration has no instances associated with it
	n_active = regressors[k].get_num_active_obs_indices()
	if n_active > 0:
    	    # Get the terms that we will need to compute term1
    	    active_inputs = regressors[k].get_active_inputs()
	    [mu, vars] = \
		regressors[k].get_target_predictions_at_inputs()

	    gram_mat = regressors[k].get_gram_mat()	    
	    inv_gram_mat = regressors[k].get_inv_gram_mat()	    
	    cov_mat = regressors[k].get_cov_mat()
	    prob_mat = regressors[k].get_prob_mat()

	    # We will need to compute the SVD of the Gram matrix in order
	    # to compute the log of its determinant. Determinants are
	    # numerically difficult to calculate. However, for sqaure
	    # positive definite and positive semi-definite matrices, the
	    # determinant is equal to the product of the singular values.
	    # Hence, the log of the Gram matrix determinant is equal to the
	    # sum of the logs of the singular values. This is a much more
	    # numerically stable way of computing this quantity.
	    [U, S, Vh] = np.linalg.svd(gram_mat)

	    # The Gram matrix must be nonsingular, otherwise we will be
	    # taking the log of 0. Check that the matrix is nonsingular
	    assert np.sum(S>0) == np.shape(S)[0], "Gram matrix is singular\
	    to working precision"

	    for d in xrange(0, D):
	        term1 += D*log(2.0*pi) + np.sum(log(S)) + \
		    np.trace(np.eye(N)-np.dot(prob_mat,cov_mat)) - \
		    np.dot(mu[:, d].T, np.dot(inv_gram_mat, mu[:, d]))

    term1 = -0.5*term1

    # Now compute term2
    term2 = 0.0

    for k in xrange(0, K):
	if regressors[k].get_num_active_obs_indices() > 0:
	    [mu, vars] = regressors[k].get_target_predictions_at_inputs()
	    for n in xrange(0, N):
		for d in xrange(0, D):
		    term2 += z_disc[n, k]*(0.5*log(2.*pi) + \
			     log(1./sqrt(beta)) + 0.5*beta*(targets[n, d]**2.\
				- 2.*targets[n, d]*mu[n, d] + mu[n, d]**2. + \
				     vars[n, d]))

    term2 *= -1.0

    # Compute term3. term3 is actually composed of two terms: one made up of
    # terms computed on the constraint subgraphs and one made up of terms
    # computed on the individual nodes
    term3 = 0.0

    # First compute the part of term3 made up of computations on the subgraphs
    for i in xrange(0, len(constraint_subgraphs)):
	expect_energy = compute_energy_expect(z_disc, constraint_subgraphs[i])
        term3 -= log(compute_partition_func(z_disc,
			regressors[0].get_gram_mat(),
		        constraint_subgraphs[i], prob_thresh)) + expect_energy

    for k in xrange(0, K):
	if regressors[k].get_num_active_obs_indices() > 0:
	    # Compute 'a'
	    a = 1.
	    for n in xrange(0, N):
		a += z_disc[n, k]

	    # Compute 'b' 
            b = alpha
	    for k2 in xrange(k+1, K):
		for n in xrange(0, N):
		    b += z_disc[n, k2]

	    # Compute digamma function values
	    psi_a = psi(a)
	    psi_b = psi(b)
	    psi_ab = psi(a+b)
	    psi_bab = 0.	    
	    for k2 in xrange(0, k-1):
		psi_bab = psi_b - psi_ab

	    # Using these terms, update 'term3'
	    for n in xrange(0, N):
		term3 += z_disc[n, k]*(psi_a - psi_ab + psi_bab)


    # Moving on to term4
    term4 = 0.

    # We need to determine how many regressors actually have a non-zero number
    # of instances assigned to them -- this is the number of significant
    # regressors
    num_sig_regressors = 0.
    
    for k in xrange(0, K):
	if regressors[k].get_num_active_obs_indices() > 0:
	    num_sig_regressors += 1.

    assert num_sig_regressors > 0, "No Gaussian Process regressor has any\
    members"
    
    term4 += num_sig_regressors*(log(gamma(1.+alpha)/gamma(alpha)))

    tmp = 0.
    for k in xrange(0, K):
	if regressors[k].get_num_active_obs_indices() > 0:    
	    # Compute 'a'
	    a = 1.
	    for n in xrange(0, N):
		a += z_disc[n, k]

	    # Compute 'b' 
            b = alpha
	    for k2 in xrange(k+1, K):
		for n in xrange(0, N):
		    b += z_disc[n, k2]

	    tmp += psi(b) - psi(a+b)

    term4 += (alpha-1.)*tmp

    # Now term5
    term5 = 0.

    # First compute the part of term5 made up of computations on the subgraphs
    for i in xrange(0, len(constraint_subgraphs)):
	expect_energy = compute_energy_expect(z_disc, constraint_subgraphs[i])
        term5 -= log(compute_partition_func(z_disc,
			regressors[0].get_gram_mat(),
		        constraint_subgraphs[i], prob_thresh)) + expect_energy

    for k in xrange(0, K):
	if regressors[k].get_num_active_obs_indices() > 0:
	    for n in xrange(0, N):
		if z_disc[n, k] > 0.:
		    term5 += z_disc[n, k]*log(z_disc_r[n, k])

    # Now term6
    term6 = 0.
    
    for k in xrange(0, K):
	if regressors[k].get_num_active_obs_indices() > 0:
    	    # Get the terms that we will need to compute term6
    	    active_inputs = regressors[k].get_active_inputs()

	    [mu, vars] = \
		regressors[k].get_target_predictions_at_inputs()		
	    cov_mat = regressors[k].get_cov_mat()
	    
	    # We'll again need to compute the SVD in order to get a stable
	    # value for the log of the determinant of the covariance matrix	
    	    [U, S, Vh] = np.linalg.svd(cov_mat)
	    
	    for d in xrange(0, D):
		term6 += -(D/2.)*log(2.*pi) - 0.5*np.sum(log(S)) - 0.5*N
		term6 += np.dot(mu[:, d].T, np.dot(pinv2(cov_mat), mu[:, d]))
  
    # Finally, compute term7
    term7 = 0.

    for k in xrange(0, K):
	if regressors[k].get_num_active_obs_indices() > 0:
	    # Compute 'a'
	    a = 1.
	    for n in xrange(0, N):
		a += z_disc[n, k]

	    # Compute 'b' 
            b = alpha
	    for k2 in xrange(k+1, K):
		for n in xrange(0, N):
		    b += z_disc[n, k2]

	    term7 += gammaln(a+b) - gammaln(a) - gammaln(b) + \
		(a-1.)*(psi(a) - psi(a+b)) + (b-1.)*(psi(b) - psi(a+b))

    # Add up all the terms to get the variational lower bound
    bound = term1 + term2 + term3 + term4 + term5 + term6 + term7
		    
    return bound

