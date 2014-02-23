import pdb

import copy
from scipy.special import psi, gamma, gammaln, polygamma, beta
from scipy.optimize import minimize_scalar
from numpy import trace, log, pi, sqrt, sum, exp, zeros, dot, diag, linspace, \
    prod, abs, outer
from scipy.linalg import inv, det, pinv2, solve, eig
import numpy as np

def dual_bp_gp_regression(data, Ks, sig_means, sig_betas,
                          noise_means, noise_betas,
                          instance_bp_param_groups,
                          feature_bp_param_groups, iters=50, K=50, repeats=0,
                          prob_thresh=0.001, phi_instances=None,
                          phi_features=None, verbose_lb=False):
    """Performs dual beta process, Gaussian Process regrssion:
    simultaneously identifies the unknown latent features, the association
    between these latent features and the instances, the association to the
    observed features). Gaussian processes are used for the likelihood term
    during inference.

    Parameters
    ----------
    data : array, shape ( N, D )
        Observed features over which to perform inference. 'N' is the number of
	instances, and 'D' is the number of observed features.

    Ks : array, shape ( D, N, N )
        Gram matrices for each of the 'D' observed features. 'N' is the number
	of instances. These matrices are used as the Gaussian Process priors.

    sig_means : array, shape ( N, D )
        Means of the GP priors for each of the 'D' observed features.

    sig_betas : array, shape ( D )
        Precision values for each of the 'D' observed features.

    noise_means : array, shape ( N, D )
        Means of the noise for each of the 'D' observed featuers. Generally,
        all elements of each vector will assume a single value.

    noise_betas : array, shape ( D )
        Noise precision values for each of the 'D' observed features.
	
    instance_bp_param_groups : list of dictionaries
        Each dictionary in the list is expected to have three keys: 'alpha'
        (float), 'gamma' (float), and 'ids' (1D array). The 'alpha' and 'gamma'
        parameters control the behaviour of the beta process prior over the
        subset of instances in the data matrix indexed by 'ids'.

    feature_bp_param_groups : list of dictionaries
        Each dictionary in the list is expected to have three keys: 'alpha'
        (float), 'gamma' (float), and 'ids' (1D array). The 'alpha' and 'gamma'
        parameters control the behaviour of the beta process prior over the
        subset of observed features in the data matrix indexed by 'ids'.

    iters : int, optional
        The number of iterations of the optimization to compute.

    K : int, optional
        Indicates the number of latent features in the approximated beta
        process

    repeats : integer, optional
        The number of times the algorithm is run with a random initialization.
	After each run, the variational lower bound is computed, and a record
	is kept of the best performing settings for the unobserved variables;
	these are the ones returned by this function.

    prob_thresh : float, optional
        The probability threshold is a scalar value in the interval (0, 1). It
        indicates the minimum value of component 'k' of the latent variable,
        'z', that a given point needs to have to be considered a possible
        member of regression curve 'k'. A value of 0.001 has been tested and
        works effectively. Setting the value too low runs the risk of numerical
        instability in the algorithm. Setting the value too high prevents
        points that may have a significant probability of belonging to a curve
        from "saying so".
    
    phi_instances : array, shape ( N, K ), optional
        Corresponds to the parameters of the binary matrix distribution
        describing the association between instances and latent features.
        'N' is the number of input points, and 'K' is the number of
        latent features used in the approximated beta process. If specified,
        the algorithm will be initialized with this matrix. Otherwise, the
        matrix will be randomly generated. Each element must be in the
        interval [0, 1].

    phi_features : array, shape ( D, K ), optional
        Corresponds to the parameters of the binary matrix distribution
        describing the association between observed features and latent
        features. 'D' is the number of observed features, and 'K' is the number
        of latent features used in the approximated beta process. If specified,
        the algorithm will be initialized with this matrix. Otherwise, the
        matrix will be randomly generated. Each element must be in the interval
        [0, 1].

    verbose_lb : bool, optional
        Set this to true in order to compute, record, and display the
        variational lower bound after every update. Running with this option
        will be significantly slower given the extra computations, but may
        be helpful to monitor convergence.

    Returns
    -------
    phi_instances : array, shape ( N, K )
        Parameters of the binary matrix distribution describing the
        association between instances and latent features. 'N' is the number of
        data instances, and 'K' is the number of latent features used in the
        approximated beta process. 

    phi_features : array, shape ( D, K ), optional
        Parameters of the binary matrix distribution describing the association
        between observed features and latent features. 'D' is the number of
        observed features, and 'K' is the number of latent features used in the
        approximated beta process.

    lower_bound : array, shape ( 1 ) or shape (iters)
        The computed variational lower bound. If 'verbose_lb' is set to True
        this will be an array of values consisting of the computed variational
        lower bound after every iteration. If 'verbose_lb' is set to False,
        this will be a single value -- the variational lower bound computed
        after all iterations have been completed. 

    References
    ----------
    1. Carin, Lawrence, David M. Blei, and John W. Paisley. "Variational
    inference for stick-breaking beta process priors." Proceedings of the 28th
    International Conference on Machine Learning (ICML-11). 2011.

    See also
    --------
    compute_kernel_func_vals : For supported kernel types.
    """
    # There are a number of latent variables in the model. We will try to
    # adhere to the variable naming conventions used in ref. [1].

    if repeats < 0:
	raise ValueError("Number of repeats must be at least 0")

    if iters < 1:
	raise ValueError("Number of iterations must be greater than 0")

    # Make sure that the param groups are set up correctly
    for i in xrange(0, len(instance_bp_param_groups)):
	assert type( instance_bp_param_groups[i] ) == dict, "Elements of \
	instance_bp_param_groups must be dictionaries"

	assert 'alpha' in instance_bp_param_groups[i], "'alpha' must be a key \
	in instance_bp_param_groups dictionaries"

	assert 'gamma' in instance_bp_param_groups[i], "'gamma' must be a key \
	in instance_bp_param_groups dictionaries"

	assert 'ids' in instance_bp_param_groups[i], "'ids' must be a key \
	in instance_bp_param_groups dictionaries"

    for i in xrange(0, len(feature_bp_param_groups)):
	assert type( feature_bp_param_groups[i] ) == dict, "Elements of \
	feature_bp_param_groups must be dictionaries"

	assert 'alpha' in feature_bp_param_groups[i], "'alpha' must be a key \
	in feature_bp_param_groups dictionaries"

	assert 'gamma' in feature_bp_param_groups[i], "'gamma' must be a key \
	in feature_bp_param_groups dictionaries"

	assert 'ids' in feature_bp_param_groups[i], "'ids' must be a key \
	in feature_bp_param_groups dictionaries"			

    R = np.max([int(K/4), 2])
    N = data.shape[0]
    D = data.shape[1]

    # Initialize if necessary
    if phi_instances is None:
	phi_instances = np.random.rand(N, K)

    # Initialize if necessary
    if phi_features is None:
	phi_features = np.random.rand(D, K)	

    # Our implementation provides for different groups of instances and/or
    # different groups of observed features to be associated with different
    # alpha and gamma parameters in order to control the behavior of the
    # beta processes over each group separately. Consequently, we will need
    # separate parameters for the other latent variables associated with the
    # beta processes in each group. In particular, we'll need params 'vphi',
    # 'phi', 'a', 'b', 'u', and 'v', corresponding to latent variables 'd',
    # 'z', 'V', and 'T' as listed in equation 9 in reference [1]. The set of
    # all parameters for the instances will be contained in a list of
    # dictionaries: all dictionaries in the list have the same keys
    # (indicating the names of the parameters), but the values will differ
    # and/or will be updated separately during the optimization. We use an
    # analogous structure for the observed features.
	
    n_instance_groups = len(instance_bp_param_groups)    
    instance_param_groups = [] # Will be the list of dictionaries

    for i in xrange(0, n_instance_groups):
	tmp = {'alpha': None, 'gamma': None, 'ids': None,
	       'a': None, 'b': None, 'u': None, 'v': None, 'phi': None,
	       'vphi': None}
	tmp['ids'] = instance_bp_param_groups[i]['ids']
	tmp['alpha'] = instance_bp_param_groups[i]['alpha']
	tmp['gamma'] = instance_bp_param_groups[i]['gamma']

	# Initialize the 'vphi' matrix. Each column represents an atom in the
	# beta process introduced in reference [1]. There are 'R' rows
	# indicating the possible "rounds" in which an atom could potentially
	# occur, and the probability of an atom occuring in of these rounds is
	# governed by a multinomial distribution. Hence, each column should sum
	# to 1 and each element of the matrix should be in the interval [0,1].	
	vphi = np.random.rand(R, K)
	for j in xrange(0, K):
	    vphi[:, j] = vphi[:, j]/sum(vphi[:, j])
	tmp['vphi'] = vphi

	tmp['phi'] = \
	    phi_instances[np.int_(instance_bp_param_groups[i]['ids']), :]
	
	tmp['a'] = np.random.rand(K)
	tmp['b'] = np.random.rand(K)
	tmp['u'] = np.random.rand(K)
	tmp['v'] = np.random.rand(K)

	instance_param_groups.append(tmp)

    # Now initialize the observed feature groups
    n_feature_groups = len(feature_bp_param_groups)    
    feature_param_groups = [] # Will be the list of dictionaries

    for i in xrange(0, n_feature_groups):
	tmp = {'alpha': None, 'gamma': None, 'ids': None,
	       'a': None, 'b': None, 'u': None, 'v': None, 'phi': None,
	       'vphi': None}
	tmp['ids'] = feature_bp_param_groups[i]['ids']
	tmp['alpha'] = feature_bp_param_groups[i]['alpha']
	tmp['gamma'] = feature_bp_param_groups[i]['gamma']

	# Initialize the 'vphi' matrix. Each column represents an atom in the
	# beta process introduced in reference [1]. There are 'R' rows
	# indicating the possible "rounds" in which an atom could potentially
	# occur, and the probability of an atom occuring in of these rounds is
	# governed by a multinomial distribution. Hence, each column should sum
	# to 1 and each element of the matrix should be in the interval [0,1].	
	vphi = np.random.rand(R, K)
	for j in xrange(0, K):
	    vphi[:, j] = vphi[:, j]/sum(vphi[:, j])
	tmp['vphi'] = vphi

	tmp['phi'] = \
	    np.atleast_2d(
		phi_features[np.int_(feature_bp_param_groups[i]['ids']), :])
	
	tmp['a'] = np.random.rand(K)
	tmp['b'] = np.random.rand(K)
	tmp['u'] = np.random.rand(K)
	tmp['v'] = np.random.rand(K)
 
	feature_param_groups.append(tmp)

    verbose_data = False
    lb_cum = -1e100
    
    # Now that we have initialized our variables, we are ready to run the
    # coordinate ascent
    lbs = [] # Will record the lower bound values
    for i in xrange(0, iters):
	mu, cov = \
	    up_F_(data, sig_means, phi_instances, phi_features, sig_betas,
                  Ks, prob_thresh)

        if verbose_lb:
            lb = lower_bound_(phi_instances = phi_instances,
		         phi_features = phi_features,
		         vphi_instances = instance_param_groups[0]['vphi'],
		         vphi_features = feature_param_groups[0]['vphi'],
		         alpha_instances = instance_param_groups[0]['alpha'],
		         alpha_features = feature_param_groups[0]['alpha'],
		         gamma_instances = instance_param_groups[0]['gamma'],
		         gamma_features = feature_param_groups[0]['gamma'],
		         a_instances = instance_param_groups[0]['a'],
		         a_features = feature_param_groups[0]['a'],
		         b_instances = instance_param_groups[0]['b'],
		         b_features = feature_param_groups[0]['b'],
		         u_instances = instance_param_groups[0]['u'],
		         u_features = feature_param_groups[0]['u'],
		         v_instances = instance_param_groups[0]['v'],
		         v_features = feature_param_groups[0]['v'],
		         mu = mu, cov = cov, Ks = Ks, \
                         sig_betas = sig_betas,
                         sig_means = sig_means,
                         noise_means = noise_means,
                         noise_betas = noise_betas,
		         y = data, prob_thresh = prob_thresh)
            lbs.append(lb)
	    print "Lb after up_F_: %s" % lb
	    	
	for j in xrange(0, len(instance_param_groups)):
	    instance_param_groups[j]['vphi'] = \
		up_d_(instance_param_groups[j]['alpha'],
		      instance_param_groups[j]['gamma'],
		      instance_param_groups[j]['vphi'],
		      instance_param_groups[j]['phi'],
		      instance_param_groups[j]['a'],
		      instance_param_groups[j]['b'],
		      instance_param_groups[j]['u'],
		      instance_param_groups[j]['v'])
            
            if verbose_lb:
                lb = lower_bound_(phi_instances = phi_instances,
		             phi_features = phi_features,
		             vphi_instances = instance_param_groups[0]['vphi'],
		             vphi_features = feature_param_groups[0]['vphi'],
		             alpha_instances = instance_param_groups[0]['alpha'],
		             alpha_features = feature_param_groups[0]['alpha'],
		             gamma_instances = instance_param_groups[0]['gamma'],
		             gamma_features = feature_param_groups[0]['gamma'],
		             a_instances = instance_param_groups[0]['a'],
		             a_features = feature_param_groups[0]['a'],
		             b_instances = instance_param_groups[0]['b'],
		             b_features = feature_param_groups[0]['b'],
		             u_instances = instance_param_groups[0]['u'],
		             u_features = feature_param_groups[0]['u'],
		             v_instances = instance_param_groups[0]['v'],
		             v_features = feature_param_groups[0]['v'],
		             mu = mu, cov = cov, Ks = Ks, \
                             sig_betas = sig_betas,
                             sig_means = sig_means,
                             noise_means = noise_means,
                             noise_betas = noise_betas,                             
		             y = data, prob_thresh = prob_thresh)
                lbs.append(lb)                

	    instance_param_groups[j]['a'], instance_param_groups[j]['b'] = \
		up_V_(instance_param_groups[j]['alpha'],
		      instance_param_groups[j]['vphi'],
		      instance_param_groups[j]['phi'],
		      instance_param_groups[j]['a'],
		      instance_param_groups[j]['b'],
		      instance_param_groups[j]['u'],
		      instance_param_groups[j]['v'])

            if verbose_lb:
                lb = lower_bound_(phi_instances = phi_instances,
		             phi_features = phi_features,
		             vphi_instances = instance_param_groups[0]['vphi'],
		             vphi_features = feature_param_groups[0]['vphi'],
		             alpha_instances = instance_param_groups[0]['alpha'],
		             alpha_features = feature_param_groups[0]['alpha'],
		             gamma_instances = instance_param_groups[0]['gamma'],
		             gamma_features = feature_param_groups[0]['gamma'],
		             a_instances = instance_param_groups[0]['a'],
		             a_features = feature_param_groups[0]['a'],
		             b_instances = instance_param_groups[0]['b'],
		             b_features = feature_param_groups[0]['b'],
		             u_instances = instance_param_groups[0]['u'],
		             u_features = feature_param_groups[0]['u'],
		             v_instances = instance_param_groups[0]['v'],
		             v_features = feature_param_groups[0]['v'],
		             mu = mu, cov = cov, Ks = Ks,
                             sig_betas = sig_betas,
                             sig_means = sig_means,
                             noise_means = noise_means,
                             noise_betas = noise_betas,                             
		             y = data, prob_thresh = prob_thresh)
                lbs.append(lb)

	    instance_param_groups[j]['u'], instance_param_groups[j]['v'] = \
		up_T_(instance_param_groups[j]['alpha'],
		      instance_param_groups[j]['vphi'],
		      instance_param_groups[j]['phi'],
		      instance_param_groups[j]['a'],
		      instance_param_groups[j]['b'],
		      instance_param_groups[j]['u'],
		      instance_param_groups[j]['v'])

            if verbose_lb:
                lb = lower_bound_(phi_instances = phi_instances,
		             phi_features = phi_features,
		             vphi_instances = instance_param_groups[0]['vphi'],
		             vphi_features = feature_param_groups[0]['vphi'],
		             alpha_instances = instance_param_groups[0]['alpha'],
		             alpha_features = feature_param_groups[0]['alpha'],
		             gamma_instances = instance_param_groups[0]['gamma'],
		             gamma_features = feature_param_groups[0]['gamma'],
		             a_instances = instance_param_groups[0]['a'],
		             a_features = feature_param_groups[0]['a'],
		             b_instances = instance_param_groups[0]['b'],
		             b_features = feature_param_groups[0]['b'],
		             u_instances = instance_param_groups[0]['u'],
		             u_features = feature_param_groups[0]['u'],
		             v_instances = instance_param_groups[0]['v'],
		             v_features = feature_param_groups[0]['v'],
		             mu = mu, cov = cov, Ks = Ks,
                             sig_betas = sig_betas,
                             sig_means = sig_means,
                             noise_means = noise_means,
                             noise_betas = noise_betas,                             
		             y = data, prob_thresh = prob_thresh)
                lbs.append(lb)

	    ids = np.int_(instance_param_groups[j]['ids'])

	    instance_param_groups[j]['phi'] = \
		up_phi_instances_(
			data[ids, :],
	                mu[ids, :, :],
	                cov[:, :, :, :][ids, :][:, ids],
	                sig_means[ids, :], sig_betas,
                        noise_means[ids, :], noise_betas,
                        phi_features,
			instance_param_groups[j]['phi'],
			instance_param_groups[j]['a'],
		        instance_param_groups[j]['b'],
		        instance_param_groups[j]['u'],
		        instance_param_groups[j]['v'],
		        instance_param_groups[j]['vphi'],
  	                prob_thresh)
	    phi_instances[ids, :] = instance_param_groups[j]['phi']

            if verbose_lb:
                lb = lower_bound_(phi_instances = phi_instances,
		             phi_features = phi_features,
		             vphi_instances = instance_param_groups[0]['vphi'],
		             vphi_features = feature_param_groups[0]['vphi'],
		             alpha_instances = instance_param_groups[0]['alpha'],
		             alpha_features = feature_param_groups[0]['alpha'],
		             gamma_instances = instance_param_groups[0]['gamma'],
		             gamma_features = feature_param_groups[0]['gamma'],
		             a_instances = instance_param_groups[0]['a'],
		             a_features = feature_param_groups[0]['a'],
		             b_instances = instance_param_groups[0]['b'],
		             b_features = feature_param_groups[0]['b'],
		             u_instances = instance_param_groups[0]['u'],
		             u_features = feature_param_groups[0]['u'],
		             v_instances = instance_param_groups[0]['v'],
		             v_features = feature_param_groups[0]['v'],
		             mu = mu, cov = cov, Ks = Ks,
                             sig_betas = sig_betas,
                             sig_means = sig_means,
                             noise_means = noise_means,
                             noise_betas = noise_betas,                         
		             y = data, prob_thresh = prob_thresh)
                lbs.append(lb)
	        print "Lb after up_phi_instances: %s" % lb

	for j in xrange(0, len(feature_param_groups)):
	    feature_param_groups[j]['vphi'] = \
		up_d_(feature_param_groups[j]['alpha'],
		      feature_param_groups[j]['gamma'],
		      feature_param_groups[j]['vphi'],
		      feature_param_groups[j]['phi'],
		      feature_param_groups[j]['a'],
		      feature_param_groups[j]['b'],
		      feature_param_groups[j]['u'],
		      feature_param_groups[j]['v'])

            if verbose_lb:
                lb = lower_bound_(phi_instances = phi_instances,
		             phi_features = phi_features,
		             vphi_instances = instance_param_groups[0]['vphi'],
		             vphi_features = feature_param_groups[0]['vphi'],
		             alpha_instances = instance_param_groups[0]['alpha'],
		             alpha_features = feature_param_groups[0]['alpha'],
		             gamma_instances = instance_param_groups[0]['gamma'],
		             gamma_features = feature_param_groups[0]['gamma'],
		             a_instances = instance_param_groups[0]['a'],
		             a_features = feature_param_groups[0]['a'],
		             b_instances = instance_param_groups[0]['b'],
		             b_features = feature_param_groups[0]['b'],
		             u_instances = instance_param_groups[0]['u'],
		             u_features = feature_param_groups[0]['u'],
		             v_instances = instance_param_groups[0]['v'],
		             v_features = feature_param_groups[0]['v'],
		             mu = mu, cov = cov, Ks = Ks,
                             sig_betas = sig_betas,
                             sig_means = sig_means,
                             noise_means = noise_means,
                             noise_betas = noise_betas,                         
		             y = data, prob_thresh = prob_thresh)
                lbs.append(lb)
	        print "Lb after up_d_ features: %s" % lb

	    feature_param_groups[j]['a'], feature_param_groups[j]['b'] = \
		up_V_(feature_param_groups[j]['alpha'],
		      feature_param_groups[j]['vphi'],
		      feature_param_groups[j]['phi'],
		      feature_param_groups[j]['a'],
		      feature_param_groups[j]['b'],
		      feature_param_groups[j]['u'],
		      feature_param_groups[j]['v'])

            if verbose_lb:
                lb = lower_bound_(phi_instances = phi_instances,
		             phi_features = phi_features,
		             vphi_instances = instance_param_groups[0]['vphi'],
		             vphi_features = feature_param_groups[0]['vphi'],
		             alpha_instances = instance_param_groups[0]['alpha'],
		             alpha_features = feature_param_groups[0]['alpha'],
		             gamma_instances = instance_param_groups[0]['gamma'],
		             gamma_features = feature_param_groups[0]['gamma'],
		             a_instances = instance_param_groups[0]['a'],
		             a_features = feature_param_groups[0]['a'],
		             b_instances = instance_param_groups[0]['b'],
		             b_features = feature_param_groups[0]['b'],
		             u_instances = instance_param_groups[0]['u'],
		             u_features = feature_param_groups[0]['u'],
		             v_instances = instance_param_groups[0]['v'],
		             v_features = feature_param_groups[0]['v'],
		             mu = mu, cov = cov, Ks = Ks,
                             sig_betas = sig_betas,
                             sig_means = sig_means,
                             noise_means = noise_means,
                             noise_betas = noise_betas,                         
		             y = data, prob_thresh = prob_thresh)
                lbs.append(lb)
	        print "Lb after up_V_ features: %s" % lb

	    feature_param_groups[j]['u'], feature_param_groups[j]['v'] = \
		up_T_(feature_param_groups[j]['alpha'],
		      feature_param_groups[j]['vphi'],
		      feature_param_groups[j]['phi'],
		      feature_param_groups[j]['a'],
		      feature_param_groups[j]['b'],
		      feature_param_groups[j]['u'],
		      feature_param_groups[j]['v'])
            
            if verbose_lb:
                lb = lower_bound_(phi_instances = phi_instances,
		             phi_features = phi_features,
		             vphi_instances = instance_param_groups[0]['vphi'],
		             vphi_features = feature_param_groups[0]['vphi'],
		             alpha_instances = instance_param_groups[0]['alpha'],
		             alpha_features = feature_param_groups[0]['alpha'],
		             gamma_instances = instance_param_groups[0]['gamma'],
		             gamma_features = feature_param_groups[0]['gamma'],
		             a_instances = instance_param_groups[0]['a'],
		             a_features = feature_param_groups[0]['a'],
		             b_instances = instance_param_groups[0]['b'],
		             b_features = feature_param_groups[0]['b'],
		             u_instances = instance_param_groups[0]['u'],
		             u_features = feature_param_groups[0]['u'],
		             v_instances = instance_param_groups[0]['v'],
		             v_features = feature_param_groups[0]['v'],
		             mu = mu, cov = cov, Ks = Ks,
                             sig_betas = sig_betas,
                             sig_means = sig_means,
                             noise_means = noise_means,
                             noise_betas = noise_betas,                         
		             y = data, prob_thresh = prob_thresh)
                lbs.append(lb)
	        print "Lb after up_T_ features: %s" % lb

	    ids = np.atleast_1d(np.int_(feature_param_groups[j]['ids']))
	    feature_param_groups[j]['phi'] = \
		up_phi_features_(
			data[:, ids],
			mu[:, :, ids],
	                cov[:, :, :, ids],
	                sig_means[:, ids], sig_betas[ids],
	                noise_means[:, ids], noise_betas[ids],                        
			feature_param_groups[j]['phi'],
			phi_instances,			
			feature_param_groups[j]['a'],
			feature_param_groups[j]['b'],
			feature_param_groups[j]['u'],
			feature_param_groups[j]['v'],
			feature_param_groups[j]['vphi'],
			prob_thresh)
	    phi_features[ids, :] = feature_param_groups[j]['phi']

            if verbose_lb or i == iters-1:
                lb = lower_bound_(phi_instances = phi_instances,
		             phi_features = phi_features,
		             vphi_instances = instance_param_groups[0]['vphi'],
		             vphi_features = feature_param_groups[0]['vphi'],
		             alpha_instances = instance_param_groups[0]['alpha'],
		             alpha_features = feature_param_groups[0]['alpha'],
		             gamma_instances = instance_param_groups[0]['gamma'],
		             gamma_features = feature_param_groups[0]['gamma'],
		             a_instances = instance_param_groups[0]['a'],
		             a_features = feature_param_groups[0]['a'],
		             b_instances = instance_param_groups[0]['b'],
		             b_features = feature_param_groups[0]['b'],
		             u_instances = instance_param_groups[0]['u'],
		             u_features = feature_param_groups[0]['u'],
		             v_instances = instance_param_groups[0]['v'],
		             v_features = feature_param_groups[0]['v'],
		             mu = mu, cov = cov, Ks = Ks,
                             sig_betas = sig_betas,
                             sig_means = sig_means,
                             noise_means = noise_means,
                             noise_betas = noise_betas,                         
		             y = data, prob_thresh = prob_thresh)
                lbs.append(lb)
                
    return phi_instances, phi_features, np.array(lbs)

def up_d_(alpha, gamma, vphi, phi, a, b, u, v):
    """Update the parameters for the variational distribution over latent
    variable d

    Parameters
    ----------
    alpha : float
        The alpha parameter controlling the beta process. See reference [1] for
	details.

    gamma : float
        The gamma paramter controlling the beta process. See reference [1] for
	details.

    vphi : array, shape ( R, K )
        Parameters of the Multinomial distribution over the latent variable
        'd'. See equation 9 in reference [1] for details. Each column
        represents an atom in the truncated beta process. Each row represents
        one of the possible "rounds" (using the terminology of [1]) in which
        the atom could have occurred.

    phi : array, shape ( L, K )
        Parameters of the Bernoulli distribution over the latent variable z_nk.
	See equation 9 in reference [1]. 'L' is the number of "entries"
	(instances or observed features, depending on how the function is
	called), and 'K' is the number of latent features in the truncated beta
        process.

    a : array, shape ( K )
        Each element corresponds to the first parameter of the Beta
	distribution for latent variable Vk. See equation 9 in reference [1].
	Here 'K' is the number of latent features in the truncated beta
	process.

    b : array, shape ( K )
        Each element corresponds to the second parameter of the Beta
	distribution for latent variable Vk. See equation 9 in reference [1].
	Here 'K' is the number of latent features in the truncated beta
	process.

    u : array, shape ( K )
        Each element corresponds to the first parameter of the Gamma
	distribution for latent variable Tk. See equation 9 in reference [1].
	Here 'K' is the number of latent features in the truncated beta
	process.

    v : array, shape ( K )
        Each element corresponds to the second parameter of the Gamma
	distribution for latent variable Tk. See equation 9 in reference [1].
	Here 'K' is the number of latent features in the truncated beta
	process.

    Returns
    -------
    vphi : array, shape ( R, K )
        Parameters of the Multinomial distribution over the latent variable
        'd'. See equation 9 in reference [1] for details. Each column
        represents an atom in the truncated beta process. Each row represents
        one of the possible "rounds" (using the terminology of [1]) in which
        the atom could have occurred.

    Notes
    -----
    Equation 18 in reference [1] provides the update equations implemented in
    this function. The implementation of this function is based on code
    provided by John Paisley. Most of the variable names have been kept the
    as in his implementation for easy comparison.

    References
    ----------
    1. Carin, Lawrence, David M. Blei, and John W. Paisley. "Variational
    inference for stick-breaking beta process priors." Proceedings of the 28th
    International Conference on Machine Learning (ICML-11). 2011.    
    """
    R = vphi.shape[0]
    K = vphi.shape[1]
    N = phi.shape[0]

    M = 1000
    xi = 0.9375
    m =  np.array(xrange(1, M+1))

    n_k = np.sum(phi, 0)

    dvarphi = np.zeros([R, K])
    tmp = np.zeros([R, K])
    for r in xrange(0, R):
        tmp[r, :] = 1. - np.sum(vphi[r:R, :], 0)

    for k in xrange(0, K):
        idx = np.delete(np.array(xrange(0, K)), k)
        for r in xrange(1, R):
	    dvarphi[r, k] = -gamma*np.sum(np.prod(tmp[0:r+1, idx], 1))

        vphi[:, k] = 0.
        tmpval = np.sum(((v[k]/(v[k] + m))**u[k])*np.exp(gammaln(a[k] + b[k]) \
          - gammaln(a[k] + b[k] + m) + gammaln(a[k] + m) - gammaln(a[k]))/m)
	
        lnvarphi = np.zeros(R)
        lnvarphi[0] = (N-n_k[k])*(psi(b[k]) - psi(a[k] + b[k])) - \
          np.sum(vphi[0, :])*xi

        HTk = u[k] - (u[k] - 1)*psi(u[k]) - log(v[k]) + gammaln(u[k])

        for r in xrange(1, R):
	    lnvarphi[r] = -n_k[k]*u[k]/v[k] - (N - n_k[k])*tmpval + \
		r*log(alpha) - gammaln(r) + (r - 1.)*(psi(u[k]) - log(v[k])) - \
		alpha*u[k]/v[k] - np.sum(vphi[r, :])*xi + dvarphi[r, k] + HTk

	lnvarphi = lnvarphi - np.max(lnvarphi)

	vphi[:, k] = np.exp(lnvarphi)
	vphi[:, k] = vphi[:, k]/np.sum(vphi[:, k])

    return vphi

def up_phi_instances_(y, mu, cov, sig_means, sig_betas,
                      noise_means, noise_betas,
                      phi_features, phi_instances,
                      a, b, u, v, vphi, prob_thresh=0.001):
    """ Update the parameters for the matrix of Bernoulli parameter values
    relating to the association between data instances and latent features

    Parameters
    ----------
    y : array, shape ( N, D )
        The observed values for each of the 'N' instances across each of the
	'D' observed featurs.
    
    mu : array, shape ( N, K, D )
        The posterior means of the GP regressors. 'N' is the number of
        instances, 'K' is the number of latent features in the truncated beta
        process, and 'D' is the number of observed features.    
	
    cov : array, shape ( N, N, K, D )
        Collection of covariance matrices for the posterior GP regressors. 'N'
	is the number of instances, 'K' is the number of latent features in the
	truncated beta process, and 'D' is the number of observed features.
	Note that the covariance matrix is only computed for those instances
	that -- for a given observed feature and latent feaure ('d' and 'k') --
	have a probability of belonging to that GP that is greater than
        'prob_thresh'. The other elements of the covariance matrix are set to
	NaN.

    sig_means : array, shape ( N, D ), optional
        Means of the GP priors for each of the 'D' observed features.

    sig_betas : array, shape ( D )
        Precision values for each of the 'D' observed features.

    noise_means : array, shape ( N, D )
        Means of the noise for each of the 'D' observed featuers. Generally,
        all elements of each vector will assume a single value.

    noise_betas : array, shape ( D )
        Noise precision values for each of the 'D' observed features.
	
    phi_features : array, shape ( D, K )
        Matrix of Bernoulli parameter values relating to the association
        between observed features and latent features. (Shape should be
        NxK if the function is used to update 'phi_features' instead of
        'phi_instances')

    phi_instances : array, shape ( N, K )
        Bernoulli parameters for the latent indicator variables. 'N' is the
        number of instances, and 'K' is the number of latent features in the
        truncated beta process. (Shape will be DxK if the function is used to
        update 'phi_features' instead of 'phi_instances')

    a : array, shape ( K )
        Each element corresponds to the first parameter of the Beta
	distribution for latent variable Vk. See equation 9 in reference [1].
	Here 'K' is the number of latent features in the truncated beta
	process.

    b : array, shape ( K )
        Each element corresponds to the second parameter of the Beta
	distribution for latent variable Vk. See equation 9 in reference [1].
	Here 'K' is the number of latent features in the truncated beta
	process.

    u : array, shape ( K )
        Each element corresponds to the first parameter of the Gamma
	distribution for latent variable Tk. See equation 9 in reference [1].
	Here 'K' is the number of latent features in the truncated beta
	process.

    v : array, shape ( K )
        Each element corresponds to the second parameter of the Gamma
	distribution for latent variable Tk. See equation 9 in reference [1].
	Here 'K' is the number of latent features in the truncated beta
	process.

    vphi : array, shape ( R, K )
        Parameters of the Multinomial distribution over the latent variable
        'd'. See equation 9 in reference [1] for details. Each column
        represents an atom in the truncated beta process. Each row represents
        one of the possible 'rounds' (using the terminology of [1]) in which
        the atom could have occurred. The probability of an atom occuring in
        of these rounds is governed by a multinomial distribution. Hence, each
        column should sum to 1 and each element of the matrix should be in the
        interval [0,1].

    prob_thresh : float
        Probability threshold. 

    Returns
    -------
    phi_instances : array, shape ( N, K )
        Updated Bernoulli parameters for the latent indicator variables. 'N'
        is the number of instances, and 'K' is the number of latent features
        in the truncated beta process. (Shape will be DxK if the function is
        used to update 'phi_features' instead of 'phi_instances')

    Notes
    -----
    Some of the variable names are meant to coincide with terms defined in
    hand-written notes: 12.12.13.1 - 12.13.13.2. These include the terms:
    d2_d0, d4_d0, d1_d1, d2_d1, and d4_d1.

    References
    ----------
    1. Carin, Lawrence, David M. Blei, and John W. Paisley. 'Variational
    inference for stick-breaking beta process priors.' Proceedings of the 28th
    International Conference on Machine Learning (ICML-11). 2011.       
    """
    N = mu.shape[0]
    D = sig_betas.shape[0]
    K = vphi.shape[1]
    M = 1000

    # Initialize the output
    phi_up = np.copy(phi_instances).astype(np.float64)

    # Pre-compute terms 't1' and 't2' (see notes on 2.2.14.1). 
    t1 = zeros([N, D])
    t2 = zeros([N, K, D])
    for d in xrange(0, D):
        b_term = 0.5*log(0.5*noise_betas[d]/pi) - \
            0.5*noise_betas[d]*(y[:, d] - noise_means[:, d])**2
        t1[:, d] = 0.5*log(0.5*sig_betas[d]/pi) - 0.5 - b_term

	# For this particular observed feature ('d') we only want to consider
	# the latent features that have a non-negligible probability of
	# being associated to them. Call these latent feature indices 'k_ids'.
	tmp =  phi_features[d, :] > prob_thresh
	k_ids = linspace(0, K-1, K).astype(int)[tmp]
	for k in k_ids:
	    # Now we only want to tally up instances that have a non-negligible
	    # probability of belonging to latent feature 'k'
	    ids = phi_instances[:, k] > prob_thresh
	    if sum(ids) > 0:
		t2[ids, k, d] = 0.5*log(K)/(K - 1) - \
		    0.5*sig_betas[d]*y[ids, d]**2 + \
		    sig_betas[d]*y[ids, d]*mu[ids, k, d]

    # Pre-compute 't3' (see notes on 2.2.14.1)
    t3 = vphi[0, :]*(psi(a[:]) - psi(b[:]))

    # Pre-compute 't4' (see notes on 2.2.14.1)
    t4 = zeros(K)
    for k in xrange(0, K):
	tmp = 0.
	for i in xrange(1, M+1):
            # The following computation uses the gamma function recursion
            # property
	    tmp += (1./i)*\
                np.prod((a[k] + linspace(0, i-1, i))/(a[k] + b[k] + \
                    linspace(0, i-1, i)))*(v[k]/(v[k] + i))**u[k]            
			
	t4[k] += (1 - vphi[0, k])*(psi(a[k]) - psi(a[k] + b[k]) - \
				   u[k]/v[k] + tmp)


    # Now that the necessary terms have been pre-computed, we can update each
    # of the elements of 'phi_instances'. Note that we only update those
    # entries that have a non-negligible probability of being associated with
    # a latent feature. 
    for k in xrange(0, K):
	for n in xrange(0, N):
	    if phi_up[n, k] > prob_thresh:
		# Now use bounded, scalar minimization to update this element
		res = minimize_scalar(fun = phi_instance_obj_,
			method = 'bounded',
                        bounds = (0., 1.),
			options = {'xtol': np.spacing(1), 'disp': False, 
				   'maxiter': 20}, 
			args = (phi_up, phi_features, mu, cov, sig_betas,
                                t1, t2, t3[k], t4[k], n, k))
                phi_up[n, k] = res.x
		
    return phi_up

def up_phi_features_(y, mu, cov, sig_means, sig_betas,
                     noise_means, noise_betas,
                     phi_features, phi_instances,
                     a, b, u, v, vphi, prob_thresh=0.001):
    """Update the parameters for the matrix of Bernoulli parameter values
    relating to the association between observed and latent features.

    Parameters
    ----------
    y : array, shape ( N, D )
        The observed values for each of the 'N' instances across each of the
	'D' observed featurs.
    
    mu : array, shape ( N, K, D )
        The posterior means of the GP regressors. 'N' is the number of
        instances, 'K' is the number of latent features in the truncated beta
        process, and 'D' is the number of observed features.    
	
    cov : array, shape ( N, N, K, D )
        Collection of covariance matrices for the posterior GP regressors. 'N'
	is the number of instances, 'K' is the number of latent features in the
	truncated beta process, and 'D' is the number of observed features.
	Note that the covariance matrix is only computed for those instances
	that -- for a given observed feature and latent feaure ('d' and 'k') --
	have a probability of belonging to that GP that is greater than
        'prob_thresh'. The other elements of the covariance matrix are set to
	NaN.

    sig_means : array, shape ( N, D ), optional
        Means of the GP priors for each of the 'D' observed features.

    sig_betas : array, shape ( D )
        Precision values for each of the 'D' observed features.

    noise_means : array, shape ( N, D )
        Means of the noise for each of the 'D' observed featuers. Generally,
        all elements of each vector will assume a single value.

    noise_betas : array, shape ( D )
        Noise precision values for each of the 'D' observed features.
	
    phi_features : array, shape ( D, K )
        Matrix of Bernoulli parameter values relating to the association
        between observed features and latent features. (Shape should be
        NxK if the function is used to update 'phi_features' instead of
        'phi_instances')

    phi_instances : array, shape ( N, K )
        Bernoulli parameters for the latent indicator variables. 'N' is the
        number of instances, and 'K' is the number of latent features in the
        truncated beta process. (Shape will be DxK if the function is used to
        update 'phi_features' instead of 'phi_instances')

    a : array, shape ( K )
        Each element corresponds to the first parameter of the Beta
	distribution for latent variable Vk. See equation 9 in reference [1].
	Here 'K' is the number of latent features in the truncated beta
	process.

    b : array, shape ( K )
        Each element corresponds to the second parameter of the Beta
	distribution for latent variable Vk. See equation 9 in reference [1].
	Here 'K' is the number of latent features in the truncated beta
	process.

    u : array, shape ( K )
        Each element corresponds to the first parameter of the Gamma
	distribution for latent variable Tk. See equation 9 in reference [1].
	Here 'K' is the number of latent features in the truncated beta
	process.

    v : array, shape ( K )
        Each element corresponds to the second parameter of the Gamma
	distribution for latent variable Tk. See equation 9 in reference [1].
	Here 'K' is the number of latent features in the truncated beta
	process.

    vphi : array, shape ( R, K )
        Parameters of the Multinomial distribution over the latent variable
        'd'. See equation 9 in reference [1] for details. Each column
        represents an atom in the truncated beta process. Each row represents
        one of the possible 'rounds' (using the terminology of [1]) in which
        the atom could have occurred. The probability of an atom occuring in
        of these rounds is governed by a multinomial distribution. Hence, each
        column should sum to 1 and each element of the matrix should be in the
        interval [0,1].

    prob_thresh : float
        Probability threshold. 

    Returns
    -------
    phi_features : array, shape ( D, K )
        Updated Bernoulli parameters for the latent indicator variables. 'D'
        is the number of observed features, and 'K' is the number of latent
        features.

    Notes
    -----
    Some of the variable names are meant to coincide with terms defined in
    hand-written notes: 2.2.14.1 - 2.3.14.1. In particular, there are four
    terms, 't1', 't2', 't3', and 't4' that are precomputed for speed purposes.

    References
    ----------
    1. Carin, Lawrence, David M. Blei, and John W. Paisley. 'Variational
    inference for stick-breaking beta process priors.' Proceedings of the 28th
    International Conference on Machine Learning (ICML-11). 2011.       
    """
    assert y.shape[0] == mu.shape[0] == cov.shape[0] == cov.shape[1] == \
    sig_means.shape[0] == phi_instances.shape[0], "Mismatch in N"

    assert y.shape[1] == mu.shape[2] == cov.shape[3] == sig_means.shape[1] == \
    phi_features.shape[0] == sig_betas.shape[0], "Mismatch in D"

    assert mu.shape[1] == cov.shape[2] == phi_features.shape[1] == \
    phi_instances.shape[1] == a.shape[0] == b.shape[0] == u.shape[0] == \
    v.shape[0] == vphi.shape[1], "Mismatch in K"

    N = mu.shape[0]
    D = sig_betas.shape[0]
    K = vphi.shape[1]
    M = 1000

    # Initialize the output
    phi_up = np.copy(phi_features).astype(np.float64)

    # Pre-compute terms 't1' and 't2' (see notes on 2.2.14.1). 
    t1 = zeros([N, D])
    t2 = zeros([N, K, D])

    for n in xrange(0, N):
        b_term = 0.5*log(0.5*noise_betas[:]/pi) - \
            0.5*noise_betas[:]*(y[n, :] - noise_means[n, :])**2
        t1[n, :] = 0.5*log(0.5*sig_betas[:]/pi) - 0.5 - b_term

	# For this particular instance ('n') we only want to consider
	# the latent features that have a non-negligible probability of
	# being associated to them. Call these latent feature indices 'k_ids'.
	tmp =  phi_instances[n, :] > prob_thresh
	k_ids = linspace(0, K-1, K).astype(int)[tmp]
	for k in k_ids:
	    # Now we only want to tally up instances that have a non-negligible
	    # probability of belonging to latent feature 'k'
	    ids = phi_features[:, k] > prob_thresh
	    if sum(ids) > 0:
		t2[n, k, ids] = 0.5*log(K)/(K - 1) - \
		    0.5*sig_betas[ids]*y[n, ids]**2 + \
		    sig_betas[ids]*y[n, ids]*mu[n, k, ids]

    # Pre-compute 't3' (see notes on 2.2.14.1)
    t3 = vphi[0, :]*(psi(a[:]) - psi(b[:]))

    # Pre-compute 't4' (see notes on 2.2.14.1)
    t4 = zeros(K)
    for k in xrange(0, K):
	tmp = 0.
	for i in xrange(1, M+1):
            # The following computation uses the gamma function recursion
            # property            
	    tmp += (1./i)*\
                np.prod((a[k] + linspace(0, i-1, i))/(a[k] + b[k] + \
                    linspace(0, i-1, i)))*(v[k]/(v[k] + i))**u[k]            
			
	t4[k] += (1 - vphi[0, k])*(psi(a[k]) - psi(a[k] + b[k]) - \
				   u[k]/v[k] + tmp)

    # Now that the necessary terms have been pre-computed, we can update each
    # of the elements of 'phi_features'. Note that we only update those
    # entries that have a non-negligible probability of being associated with
    # a latent feature. 
    for k in xrange(0, K):
	for d in xrange(0, D):
	    if phi_up[d, k] > prob_thresh:
		# Now use bounded, scalar minimization to update this element
		res = minimize_scalar(fun = phi_feature_obj_,
			method = 'bounded',
                        bounds = (0., 1.),                        
			options = {'xtol': np.spacing(1), 'disp': False, 
				   'maxiter': 20}, 
			args = (phi_instances, phi_up, mu, cov, sig_betas,
                                t1, t2, t3[k], t4[k], d, k))
		phi_up[d, k] = res.x

    return phi_up

def up_V_(alpha, vphi, phi, a, b, u, v, iters=5):
    """Update the parameters for the variational distribution over latent
    variable 'V'

    Parameters
    ----------
    alpha : float
        The alpha parameter controlling the beta process. See reference [1] for
	details.

    vphi : array, shape ( R, K )
        Parameters of the Multinomial distribution over the latent variable
        'd'. See equation 9 in reference [1] for details. Each column
        represents an atom in the truncated beta process. Each row represents
        one of the possible "rounds" (using the terminology of [1]) in which
        the atom could have occurred.

    phi : array, shape ( L, K )
        Parameters of the Bernoulli distribution over the latent variable z_nk.
	See equation 9 in reference [1]. 'L' is the number of "entries"
	(instances or observed features, depending on how the function is
	called), and 'K' is the number of latent features in the truncated beta
        process.

    a : array, shape ( K )
        Each element corresponds to the first parameter of the Beta
	distribution for latent variable Vk. See equation 9 in reference [1].
	Here 'K' is the number of latent features in the truncated beta
	process.

    b : array, shape ( K )
        Each element corresponds to the second parameter of the Beta
	distribution for latent variable Vk. See equation 9 in reference [1].
	Here 'K' is the number of latent features in the truncated beta
	process.

    u : array, shape ( K )
        Each element corresponds to the first parameter of the Gamma
	distribution for latent variable Tk. See equation 9 in reference [1].
	Here 'K' is the number of latent features in the truncated beta
	process.

    v : array, shape ( K )
        Each element corresponds to the second parameter of the Gamma
	distribution for latent variable Tk. See equation 9 in reference [1].
	Here 'K' is the number of latent features in the truncated beta
	process.

    iters : integer, optional
        The number of gradient ascent iterations to execute
	
    Returns
    -------
    a : float, shape ( K )
        Updated (first) parameter for the Beta distribution over latent
        variable Vk (see equation 9 in reference [1]).

    b : float, shape ( K )
        Updated (second) parameter for the Beta distribution over latent
        variable Vk (see equation 9 in reference [1]).

    Notes
    -----
    Equation 19 in reference [1] provides the update equations implemented in
    this function. The implementation of this function is based on code
    provided by John Paisley. Most of the variable names have been kept the
    as in his implementation for easy comparison.

    References
    ----------
    1. Carin, Lawrence, David M. Blei, and John W. Paisley. 'Variational
    inference for stick-breaking beta process priors.' Proceedings of the 28th
    International Conference on Machine Learning (ICML-11). 2011.        
    """
    M = 1000
    K = vphi.shape[1]
    N = phi.shape[0]
    m =  np.array(xrange(1, M+1))
    n_k = np.sum(phi, 0)

    for i in xrange(0, iters): 
	for k in xrange(0, K):
            tmpval = (v[k]/(v[k]+m))**u[k]*np.exp(gammaln(a[k]+b[k]) - \
		gammaln(a[k]+b[k]+m) + gammaln(a[k]+m)-gammaln(a[k]))/m
            tmpval_ak = np.sum((psi(a[k]+b[k]) + psi(a[k]+m) - \
				psi(a[k]+b[k]+m) - psi(a[k]))*tmpval)
            tmpval_bk = np.sum((psi(a[k] + b[k]) - \
				psi(a[k] + b[k] + m))*tmpval)
            dak = (n_k[k] + 1 - a[k])*(polygamma(1, a[k]) - \
		polygamma(1, a[k] + b[k])) - ((N-n_k[k])*vphi[0, k] + alpha - \
			b[k])*polygamma(1, a[k] + b[k]) - \
			(N-n_k[k])*np.sum(vphi[1::, k])*tmpval_ak
	    dbk = - (n_k[k] + 1 - a[k])*polygamma(1, a[k]+b[k]) + \
		((N-n_k[k])*vphi[0, k] + alpha-b[k])*(polygamma(1, b[k]) - \
		polygamma(1, a[k]+b[k])) - (N-n_k[k])*(1-vphi[0, k])*tmpval_bk	    
	    stepsize = get_step_V_(np.array([dak, dbk]), \
		np.array([a[k], b[k]]), alpha, n_k[k], \
		    vphi[:, k], N, u[k], v[k])
	    a[k] = a[k] + stepsize*dak
	    b[k] = b[k] + stepsize*dbk
    
    return a, b

def up_T_(alpha, vphi, phi, a, b, u, v, iters=5):
    """Update the parameters for the variational distribution over latent
    variable 'T'

    alpha : float
        The alpha parameter controlling the beta process. See reference [1] for
	details.

    vphi : array, shape ( R, K )
        Parameters of the Multinomial distribution over the latent variable
        'd'. See equation 9 in reference [1] for details. Each column
        represents an atom in the truncated beta process. Each row represents
        one of the possible "rounds" (using the terminology of [1]) in which
        the atom could have occurred.

    phi : array, shape ( L, K )
        Parameters of the Bernoulli distribution over the latent variable z_nk.
	See equation 9 in reference [1]. 'L' is the number of "entries"
	(instances or observed features, depending on how the function is
	called), and 'K' is the number of latent features in the truncated beta
        process.

    a : array, shape ( K )
        Each element corresponds to the first parameter of the Beta
	distribution for latent variable Vk. See equation 9 in reference [1].
	Here 'K' is the number of latent features in the truncated beta
	process.

    b : array, shape ( K )
        Each element corresponds to the second parameter of the Beta
	distribution for latent variable Vk. See equation 9 in reference [1].
	Here 'K' is the number of latent features in the truncated beta
	process.

    u : array, shape ( K )
        Each element corresponds to the first parameter of the Gamma
	distribution for latent variable Tk. See equation 9 in reference [1].
	Here 'K' is the number of latent features in the truncated beta
	process.

    v : array, shape ( K )
        Each element corresponds to the second parameter of the Gamma
	distribution for latent variable Tk. See equation 9 in reference [1].
	Here 'K' is the number of latent features in the truncated beta
	process.

    iters : integer, optional
        The number of gradient ascent iterations to execute
	
    Returns
    -------
    u : float, shape ( K )
        Updated (first) parameter for the Gamma distribution over latent
        variable Tk (see equation 9 in reference [1]).

    v : float, shape ( K )
        Updated (second) parameter for the Gamma distribution over latent
        variable Tk (see equation 9 in reference [1]).

    Notes
    -----
    Equation 20 in reference [1] provides the update equations implemented in
    this function. The implementation of this function is based on code
    provided by John Paisley. Most of the variable names have been kept the
    as in his implementation for easy comparison.

    References
    ----------
    1. Carin, Lawrence, David M. Blei, and John W. Paisley. 'Variational
    inference for stick-breaking beta process priors.' Proceedings of the 28th
    International Conference on Machine Learning (ICML-11). 2011.        
    """
    M = 1000
    R = vphi.shape[0]    
    K = vphi.shape[1]
    N = phi.shape[0]
    m =  np.array(xrange(1, M+1))
    n_k = np.sum(phi, 0)	

    for stepnum in xrange(0, iters): 
	for k in xrange(0, K): 
	    if vphi[0, k] > 1 - 10e-6:
		vphi[0, k] = 1 - 10e-6
		vphi[1, k] = 10e-6
		
	    tmpval = (v[k]/(v[k] + m))**u[k]*np.exp(gammaln(a[k] + b[k]) - \
						    gammaln(a[k] + b[k] + m) +\
						    gammaln(a[k] + m) - \
						    gammaln(a[k]))/m
	    tmpval_uk = np.sum(log(v[k]/(v[k] + m))*tmpval)
	    tmpval_vk = np.sum((u[k]/v[k] - u[k]/(v[k] + m))*tmpval)
	    duk = (1-vphi[0, k])*(1 - (n_k[k] + alpha)/v[k] - \
		(N-n_k[k])*tmpval_uk + (1-u[k])*polygamma(1, u[k])) + \
		polygamma(1, u[k])*np.dot(vphi[1::, k], \
					  np.array(xrange(2, R+1)) - 2)
	    dvk = (1-vphi[0, k])*((n_k[k] + alpha)*u[k]/(v[k]**2) - \
		(N-n_k[k])*tmpval_vk - 1/v[k]) - \
		(1/v[k])*np.dot(vphi[1::, k], np.array(xrange(2, R+1)) - 2)	    
	    stepsize = get_step_T_(np.array([duk, dvk]), np.array([u[k], v[k]]),
	    			   alpha, n_k[k], vphi[:, k], N, R, a[k], b[k])	    

	    u[k] = u[k] + stepsize*duk
	    v[k] = v[k] + stepsize*dvk
	    v[np.isnan(v)] = np.spacing(1)
	    v[v < np.spacing(1)] = np.spacing(1)

    return u, v

def up_F_(y, sig_means, phi_instances, phi_features, sig_betas, Ks, \
          prob_thresh=0.001):
    """Update the parameters for the variational distribution over latent
    variable F

    Note that the algorithm only needs the means of the latent functions, so
    only these will be computed / returned.

    Parameters
    ----------
    y : array, shape ( N, D )
        The observed values for each of the 'N' instances across each of the
	'D' observed featurs.
    
    sig_means : array, shape ( N, D )
        Means of the GP priors for each of the 'D' observed features.

    phi_instances : array, shape ( N, K )
        Matrix of Bernoulli parameter values relating to the association
        between instances and latent features.

    phi_features : array, shape ( D, K )
        Matrix of Bernoulli parameter values relating to the association
        between observed features and latent features.

    sig_betas : array, shape ( D )
        Precision values for each of the 'D' observed features.

    Ks : array, shape ( D, N, N )
        Gram matrices for each of the 'D' observed features. 'N' is the number
	of instances.

    prob_thresh : float
        Probability threshold. Means for the kth latent function for feature
	'd' won't be updated unless 'phi_features[d][k]' is greater than this
	threshold.

    Returns
    -------
    mu : array, shape ( N, K, D )
        The posterior means of the GP regressors. 'N' is the number of
        instances, 'K' is the number of latent features in the truncated beta
        process, and 'D' is the number of observed features.

    cov : array, shape ( N, N, K, D )
        Collection of covariance matrices for the posterior GP regressors. 'N'
	is the number of instances, 'K' is the number of latent features in the
	truncated beta process, and 'D' is the number of observed features.
	Note that the covariance matrix is only computed for those instances
	that -- for a given observed feature and latent feaure ('d' and 'k') --
	have a probability of belonging to that GP that is greater than
        'prob_thresh'. The other elements of the covariance matrix are set to
	NaN.
    """
    assert phi_instances.shape[1] == phi_features.shape[1], "Number of latent \
    functions in instance and feature arrays is not equal"
    assert sig_means.shape[0] == phi_instances.shape[0], "Mismatch between \
    instance Bernoulli parameter matrix and means matrix"
    assert sig_means.shape[1] == phi_features.shape[0], \
    "Mismatch between feature Bernoulli parameter matrix and means matrix"
    assert sig_betas.shape[0] == sig_means.shape[1], "Mismatch between means \
    matrix and the number of precision values"
    assert Ks.shape[1] == Ks.shape[2] == sig_means.shape[0], \
    "Instance mismatch within gram matrices and means matrix"
    assert Ks.shape[0] == phi_features.shape[0], "Mismatch between number of \
    Gram matrices and number of observed features"
    assert y.shape[0] == sig_means.shape[0], "Number of instances for observed\
    feature values is not equal to the number of instances for the GP means"
    assert y.shape[1] == sig_means.shape[1], "Number of  observed feature \
    values is not equal to the number of GP mean vectors"

    N = y.shape[0]
    K = phi_instances.shape[1]
    D = phi_features.shape[0]

    ep_mat = ep_mat_(phi_instances, phi_features)

    # Allocate space for the output
    mu = np.zeros([N, K, D])
    mu[:, :, :] = np.NAN
    cov = zeros([N, N, K, D])
    cov[:, :, :, :] = np.NAN

    for k in xrange(0, K):
    	for d in xrange(0, D):
            if phi_features[d, k] > prob_thresh:
    		ids = phi_instances[:, k] > prob_thresh
		n_ids = sum(ids)
		ids_flattened = np.outer(ids, ids).reshape(N*N)
		
    	        R = sig_betas[d]*np.diag(ep_mat[ids, d]*phi_instances[ids, k]*\
				     phi_features[d, k])
    	        invR = np.diag(np.diag(R)**-1)

                mu[ids, k, d] = \
		    dot(invR, solve(Ks[d, :, ids][:, ids] + invR, \
                                    sig_means[ids, d] + \
			dot(np.dot(Ks[d, :, ids][:, ids], R), y[ids, d])))

                tmp = dot(invR, solve(Ks[d, :, ids][:, ids] + \
                          invR, Ks[d, :, ids][:, ids]))

		cov[:, :, k, d].reshape(N*N)[ids_flattened] = \
		    tmp.reshape(n_ids*n_ids)
		
    return mu, cov

def get_step_V_(grad, curr, alpha, n_k, vphi, N, u, v):
    """
    Find the step size for the gradient ascent used to update the latent
    variable 'V'.

    Parameters
    ----------
    grad : array, shape ( 2 )
        The gradient calculated using equation 19 in ref. [1]

    curr : array, shape ( 2)
        The current values of the parameters 'a' and 'b' describing the Beta
        distribution over latent variable 'V'

    alpha : float
        The alpha parameter controlling the beta process. See reference [1] for
	details.

    n_k : float
        The kth element of the vector obtained by summing the 'phi' matrix (the
        matrix of Bernoulli param values over latent variable 'z') along the
        'entries' direction.

    vphi : array, shape ( R )
        The kth column of the Multinomial distribution over the latent variable
        'd'. See equation 9 in reference [1] for details. 

    N : integer
        The number of "entries" (instances or observed features, depending on
        how the function is called)

    u : float
        The first parameter of the Gamma distribution for latent variable Tk.
        See equation 9 in reference [1].

    v : float
        The second parameter of the Gamma distribution for latent variable Tk.
        See equation 9 in reference [1].

    Returns
    -------
    step : float
        The stepsize to use during the gradient ascent described in equation 19
        in reference [1].

    Notes
    -----
    The implementation of this function is based on code provided by
    John Paisley. Most of the variable names have been kept the as in his
    implementation for easy comparison. The step size is used with the
    equations 19 in reference [1].

    References
    ----------
    1. Carin, Lawrence, David M. Blei, and John W. Paisley. "Variational
    inference for stick-breaking beta process priors." Proceedings of the 28th
    International Conference on Machine Learning (ICML-11). 2011.    

    """
    M = 1000
    m =  np.array(xrange(1, M+1))

    bound = -curr/grad

    isbound = True
    if np.sum(bound > 0) == 0:
	isbound = False
	maxstep = 1.0
    else:
	maxstep = np.min(bound[bound > 0.])

    stepcheck = maxstep*np.array([.125, .25, .375, .5, .625, .75, .875, .99])

    f = -np.inf*np.ones(stepcheck.shape[0])
    for s in reversed(xrange(0, stepcheck.shape[0])):
	a = curr[0] + stepcheck[s]*grad[0]
	b = curr[1] + stepcheck[s]*grad[1]
	tmpval = np.sum(((v/(v + m))**u)*np.exp(gammaln(a + b) + \
		gammaln(a + m) - gammaln(a + b + m) - gammaln(a))/m)
	f[s] = n_k*(psi(a) - psi(a+b)) + (N-n_k)*vphi[0]*(psi(b) - psi(a+b)) - \
	    (N-n_k)*np.sum(vphi[1::])*tmpval + (alpha-1.)*(psi(b) - psi(a+b)) -\
	    (gammaln(a+b) - gammaln(a) - gammaln(b) + (a-1.)*(psi(a) - \
					psi(a+b)) + (b-1.)*(psi(b) - psi(a+b)))

	if s < stepcheck.shape[0]-2:
	    if f[s+1] > f[s]:
		break

    maxf = np.max(f)
    t2 = np.nonzero(f == maxf)[0][0]
    stepsize = stepcheck[t2]

    if t2 == 0:
	bool = True
	while bool:
	    rho = .5
	    stepsize = rho*stepsize
	    a = curr[0] + stepsize*grad[0]
	    b = curr[1] + stepsize*grad[1]
	    tmpval = np.sum((v/(v+m))**u*np.exp(gammaln(a+b) + gammaln(a+m) - \
						gammaln(a+b+m) - gammaln(a))/m)

	    fnew = n_k*(psi(a) - psi(a+b)) + (N-n_k)*vphi[0]*(psi(b)-psi(a+b)) \
		- (N-n_k)*np.sum(vphi[1::])*tmpval + (alpha-1)*(psi(b) - \
			psi(a+b)) - (gammaln(a+b) - gammaln(a) - gammaln(b) + \
				(a-1)*(psi(a) - psi(a+b)) + (b-1)*(psi(b) - \
								   psi(a+b)))

	    if fnew > maxf:
		maxf = fnew
	    else:		    
		stepsize = stepsize/rho
                break 

    return stepsize

def get_step_T_(grad, curr, alpha, n_k, vphi, N, R, a, b):
    """
    Find the step size for the gradient ascent used to update the latent
    variable 'T'.

    Parameters
    ----------
    grad : array, shape ( 2 )
        The gradient calculated using equation 20 in ref. [1]

    curr : array, shape ( 2)
        The current values of the parameters 'u' and 'v' describing the Gamma
        distribution over latent variable 'T'

    alpha : float
        The alpha parameter controlling the beta process. See reference [1] for
	details.

    n_k : float
        The kth element of the vector obtained by summing the 'phi' matrix (the
        matrix of Bernoulli param values over latent variable 'z') along the
        'entries' direction.

    vphi : array, shape ( R )
        The kth column of the Multinomial distribution over the latent variable
        'd'. See equation 9 in reference [1] for details. 

    N : integer
        The number of "entries" (instances or observed features, depending on
        how the function is called)

    R : integer
        The number of "rounds" in the stick-breaking beta process described in
	reference [1].

    a : float    
        The first parameter of the Beta distribution for latent variable Vk.
	See equation 9 in reference [1].
	
    b : float
        The second parameter of the Beta distribution for latent variable Vk.
	See equation 9 in reference [1].

    Returns
    -------
    step : float
        The stepsize to use during the gradient ascent described in equation 20
        in reference [1].

    Notes
    -----
    The implementation of this function is based on code provided by
    John Paisley. Most of the variable names have been kept the as in his
    implementation for easy comparison. The step size is used with the
    equations 20 in reference [1].

    References
    ----------
    1. Carin, Lawrence, David M. Blei, and John W. Paisley. "Variational
    inference for stick-breaking beta process priors." Proceedings of the 28th
    International Conference on Machine Learning (ICML-11). 2011.    
    """
    M = 1000
    m =  np.array(xrange(1, M+1))

    bound = -curr/grad

    isbound = True
    if np.sum(bound > 0) == 0:
	isbound = False
	maxstep = 1.0
    else:
	maxstep = np.min(bound[bound > 0.])

    stepcheck = maxstep*np.array([.125, .25, .375, .5, .625, .75, .875, .99])

    f = -np.inf*np.ones(stepcheck.shape[0])
    for s in reversed(xrange(0, stepcheck.shape[0])):
	u = curr[0] + stepcheck[s]*grad[0]
	v = curr[1] + stepcheck[s]*grad[1]
	tmpval = np.sum(((v/(v + m))**u)*np.exp(gammaln(a + b) + \
		gammaln(a + m) - gammaln(a + b + m) - gammaln(a))/m)

	f[s] = -n_k*np.sum(vphi[1::])*u/v - (N-n_k)*np.sum(vphi[1::])*tmpval +\
	    (psi(u) - log(v))*np.dot(vphi[1::],  np.array(xrange(2,R+1))-2) - \
	    alpha*np.sum(vphi[1::])*u/v + np.sum(vphi[1::])*(u - \
					(u-1)*psi(u) - log(v) + gammaln(u))
	
	if s < stepcheck.shape[0]-2:
	    if f[s+1] > f[s]:
		break

    maxf = np.max(f)
    t2 = np.nonzero(f == maxf)[0][0]
    stepsize = stepcheck[t2]

    if t2 == 0:
	bool = True
	while bool:
	    rho = .5
	    stepsize = rho*stepsize
	    u = curr[0] + stepsize*grad[0]
	    v = curr[1] + stepsize*grad[1]
	    tmpval = np.sum((v/(v+m))**u*np.exp(gammaln(a+b) + gammaln(a+m) - \
						gammaln(a+b+m) - gammaln(a))/m)
	    fnew = - n_k*np.sum(vphi[1::])*u/v - \
		(N-n_k)*np.sum(vphi[1::])*tmpval + \
		(psi(u)-log(v))*np.dot(vphi[1::],  np.array(xrange(2,R+1))-2) -\
		alpha*np.sum(vphi[1::])*u/v + \
		np.sum(vphi[1::])*(u - (u-1)*psi(u) - log(v) + gammaln(u))

	    if fnew > maxf:
		maxf = fnew
	    else:		    
		stepsize = stepsize/rho
                break 

    return stepsize

def lower_bound_(phi_instances=None, phi_features=None,
		 vphi_instances=None, vphi_features=None,
		 alpha_instances=None, alpha_features=None,
		 gamma_instances=None, gamma_features=None,
		 a_instances=None, a_features=None,
		 b_instances=None, b_features=None,
		 u_instances=None, u_features=None,
                 v_instances=None, v_features=None,
		 mu=None, cov=None, Ks=None,
                 sig_means=None, sig_betas=None,
                 noise_means=None, noise_betas=None,                 
		 y=None, prob_thresh=0.001):
    """Computes the variational lower bound

    Parameters
    ----------
    phi_instances : array, shape ( N, K ), optional
        Matrix of Bernoulli parameter values relating to the association
        between instances and latent features.

    phi_features : array, shape ( D, K ), optional
        Matrix of Bernoulli parameter values relating to the association
        between observed features and latent features.

    vphi_instances : array, shape ( R, K ), optional
        Parameters of the Multinomial distribution over the latent variable
        'd'. See equation 9 in reference [1] for details. Each column
        represents an atom in the truncated beta process relating to the
	assocation between data instances and latent features. Each row
        represents one of the possible 'rounds' (using the terminology of [1])
	in which the atom could have occurred.

    vphi_features : array, shape ( R, K ), optional
        Parameters of the Multinomial distribution over the latent variable
        'd'. See equation 9 in reference [1] for details. Each column
        represents an atom in the truncated beta process relating to the
        association between observed features and latent features. Each row
        represents one of the possible 'rounds' (using the terminology of [1])
        in which the atom could have occurred.

    alpha_instances : float, optional
        The alpha parameter controlling the beta process influencing the
        association between instances and latent features. See reference [1]
        for details.

    alpha_featuers : float, optional
        The alpha parameter controlling the beta process influencing the
        association between observed and latent features. See reference [1]
        for details.

    a_instances : array, shape ( K ), optional
        The first parameter of the Beta distribution for latent variable Vk
        (instances). See equation 9 in reference [1].

    a_features : array, shape ( K ), optional
        The first parameter of the Beta distribution for latent variable Vk
        (observed features). See equation 9 in reference [1].
	
    b_instances : array, shape ( K ), optional
        The second parameter of the Beta distribution for latent variable Vk
        (instances). See equation 9 in reference [1].

    b_features : array, shape ( K ), optional
        The second parameter of the Beta distribution for latent variable Vk
        (features). See equation 9 in reference [1].

    u_instances : array, shape ( K ), optional
        The first parameter of the Gamma distribution for latent variable Tk
        (instances). See equation 9 in reference [1].

    u_features : array, shape ( K ), optional
        The first parameter of the Gamma distribution for latent variable Tk
        (features). See equation 9 in reference [1].

    v_instances : array, shape ( K ), optional
        The second parameter of the Gamma distribution for latent variable Tk
        (instances). See equation 9 in reference [1].

    v_features : array, shape ( K ), optional
        The second parameter of the Gamma distribution for latent variable Tk
        (featuers). See equation 9 in reference [1].

    mu : array, shape ( N, K, D ), optional
        The posterior means of the GP regressors. 'N' is the number of
        instances, 'K' is the number of latent features in the truncated beta
        process, and 'D' is the number of observed features.    

    cov : array, shape ( N, N, K, D )
        Collection of covariance matrices for the posterior GP regressors. 'N'
	is the number of instances, 'K' is the number of latent features in the
	truncated beta process, and 'D' is the number of observed features.
	Note that the covariance matrix is only computed for those instances
	that -- for a given observed feature and latent feaure ('d' and 'k') --
	have a probability of belonging to that GP that is greater than
        'prob_thresh'. The other elements of the covariance matrix are set to
	NaN.

    Ks : array, shape ( D, N, N ), optional
        Gram matrices for each of the 'D' observed features. 'N' is the number
	of instances.

    sig_means : array, shape ( N, D ), optional
        Means of the GP priors for each of the 'D' observed features.

    sig_betas : array, shape ( D ), optional
        Precision values for each of the 'D' observed features.

    noise_means : array, shape ( N, D )
        Means of the noise for each of the 'D' observed featuers. Generally,
        all elements of each vector will assume a single value.

    noise_betas : array, shape ( D )
        Noise precision values for each of the 'D' observed features.

    y : array, shape ( N, D ), optional
        The observed values for each of the 'N' instances across each of the
	'D' observed featurs.	

    prob_thresh : float, optional
        Probability threshold. 

    Returns
    -------
    lower_bound : float
        The variational lower bound

    Notes
    -----
    All input parameters have been made optional in order to test various
    components of the lower bound. If not testing, all inputs should be
    used.

    References
    ----------
    1. Carin, Lawrence, David M. Blei, and John W. Paisley. 'Variational
    inference for stick-breaking beta process priors.' Proceedings of the 28th
    International Conference on Machine Learning (ICML-11). 2011.
    """
    # There are a number of terms that comprise the lower bound. Dedicated
    # code blocks compute each one, and the results are added together for the
    # final result.

    M = 1000

    if phi_instances is not None and phi_features is not None:
        ep_mat = ep_mat_(phi_instances, phi_features)	

    lb_7_7 = 0.0
    lb_1_1 = 0.0
    lb_1_2 = 0.0
    if Ks is not None and sig_betas is not None and phi_features is not None \
	and prob_thresh is not None and mu is not None and cov is not None \
	and phi_instances is not None and noise_means is not None \
        and noise_betas is not None:
	N = mu.shape[0]
	K = mu.shape[1]
	D = mu.shape[2]

	# Compute lb_1_2. It's a bit brute force to redo loops here, but this
	# term has been singled out because of it's complexity. 
	for n in xrange(0, N):
	    for d in xrange(0, D):
		# Compute b_term. The variable name is used to correspond to
		# notes on page 1.30.14.3.
		b_term = 0.5*log(0.5*noise_betas[d]/pi) - \
		    0.5*noise_betas[d]*(y[n, d] - noise_means[n, d])**2

		tmp = 0.0
		for k in xrange(0, K):
		    tmp += phi_instances[n, k]*phi_features[d, k]*\
			(0.5*log(K)/(K - 1) - 0.5*sig_betas[d]*y[n, d]**2)

		    if not np.isnan(mu[n, k, d]):
			tmp += phi_instances[n, k]*phi_features[d, k]*\
			    (sig_betas[d]*y[n, d]*mu[n, k, d])

                ids = np.logical_not(np.isnan(mu[n, :, d]))
                expec = 0.0
                if sum(ids) > 0:
                    expec = \
                        expec_zf_sqr(phi_instances[n, ids], \
                                     phi_features[d, ids], \
                                     mu[n, ids, d], cov[n, n, ids, d])
		lb_1_2 += ep_mat[n, d]*(tmp - 0.5*log(0.5*sig_betas[d]/pi) - \
                            0.5 - 0.5*sig_betas[d]*expec - b_term) + b_term

	assert D == sig_betas.shape[0], "Mismatch between number of precision \
        terms and number of observed features in GP means"
	assert D == phi_features.shape[0], "Mismatch between number of observed \
	features in GP means and number of observed features in phi_features"
	assert sig_means.shape[1] == D, "Mismatch between number of features \
        in GP means and number of features in posterior GP means"
	assert sig_means.shape[0] == N, "Mismatch between number of instances \
        in GP means and number of instances in posterior GP means"

        for k in xrange(0, K):
	    for d in xrange(0, D):
	        # We only want to update the lower bound for this particular
		# latent feature provided that the observed feature has a
		# reasonable chance of being associated with it
	        if phi_features[d, k] > prob_thresh:
                    # Identify the means that are actually defined
                    ids = np.logical_not(np.isnan(mu[:, k, d]))

    	            Rmat = sig_betas[d]*np.diag(ep_mat[ids, d]*\
			phi_instances[ids, k]*phi_features[d, k])
	            invR = np.diag(np.diag(Rmat)**-1)

		    # Note that we try to use solvers as much as possible
		    # instead of computing inverses directly, but in this case
		    # K can be poorly conditioned, so we opt for computing the
		    # pseud-inverse
		    invK = pinv2(Ks[d, :, ids][:, ids])

	            if sum(ids) > 0:
		        lb_1_1 -= 0.5*(trace(solve(Ks[d, :, ids][:, ids] + \
				invR, invR)) - np.dot(mu[ids, k, d], \
				np.dot(invK, mu[ids, k, d] + \
                                       2*sig_means[ids, d])))

	            if sum(ids) > 0:
			# The following is one half the log of the determinant
			# of the covariance matrix. It's a covariance matrix,
			# so it should theoretically be positive definite. It
			# may computationally be found not to be (or be close
			# singular), so we need to guard against taking the
			# log of a number that is too small. This is why we
			# clip at 1e-300.
	                lb_7_7 += \
			    0.5*sum(log(abs(\
				eig(cov[:, :, k, d][ids, :][:, ids])[0]).\
				clip(1e-300)))

    lb_3_instances = 0.0
    lb_4_instances = 0.0
    lb_8_instances = 0.0
    if phi_instances is not None and vphi_instances is not None and \
	alpha_instances is not None and \
	u_instances is not None and v_instances is not None and \
	a_instances is not None and b_instances is not None:

	R = vphi_instances.shape[0]
	K = vphi_instances.shape[1]
	
	Rs = np.linspace(1, R, R)

        for k in xrange(0, K):
    	    # Compute the expected value of d for this column of vphi
            exp_d_instances = \
		np.dot(vphi_instances[:, k], np.linspace(1, R, R))

	    lb_3_instances += log(alpha_instances)*np.dot(Rs[1::]-1, \
		vphi_instances[1::, k]) - np.dot(gammaln(Rs[1::]-1), \
		vphi_instances[1::, k]) + (psi(u_instances[k]) - \
		log(v_instances[k]))*np.dot(Rs[1::]-2, vphi_instances[1::, k]) \
		- alpha_instances*(1 - vphi_instances[0, k])*\
		u_instances[k]/v_instances[k]
	
	    # Update 'lb_4_instances'. 'tmp1' and 'tmp2' are constituent terms
	    tmp1 = psi(a_instances[k]) - psi(a_instances[k] + \
					     b_instances[k]) - \
					     u_instances[k]/v_instances[k]

	    tmp2 = 0.0
	    for i in xrange(1, M+1):
	        tmp2 -= (1./i)*gamma_ratios_(a_instances[k] + b_instances[k],
		    a_instances[k], i)*\
                    (v_instances[k]/(v_instances[k] + i))**u_instances[k]

	    lb_4_instances += \
		(1 - vphi_instances[0, k])*(sum(phi_instances[:, k])*tmp1 + \
				            sum(1 - phi_instances[:, k])*tmp2)

            # Update 'lb_8_instances'. This is the last term in equation 10 of
	    # ref [1] (in my notes, page 11.8.13.4). 
	    lb_8_instances += (1-vphi_instances[0, k])*(gammaln(u_instances[k]) - \
	    	    (u_instances[k] - 1)*psi(u_instances[k]) - \
		    log(v_instances[k]) + u_instances[k])

    lb_3_features = 0.0
    lb_4_features = 0.0
    lb_8_features = 0.0
    if phi_features is not None and vphi_features is not None and \
	alpha_features is not None and \
	u_features is not None and v_features is not None and \
	a_features is not None and b_features is not None:

	R = vphi_features.shape[0]
	K = vphi_features.shape[1]
	    
	Rs = np.linspace(1, R, R)

        for k in xrange(0, K):
    	    # Compute the expected value of d for this column of vphi
            exp_d_features = \
		np.dot(vphi_features[:, k], np.linspace(1, R, R))
	
	    lb_3_features += log(alpha_features)*np.dot(Rs[1::]-1, \
		vphi_features[1::, k]) - np.dot(gammaln(Rs[1::]-1), \
		vphi_features[1::, k]) + (psi(u_features[k]) - \
		log(v_features[k]))*np.dot(Rs[1::]-2, vphi_features[1::, k]) \
		- alpha_features*(1 - vphi_features[0, k])*\
		u_features[k]/v_features[k]
	    
	    # Now update 'lb_4_features'. 'tmp1' and 'tmp2' are constituent
	    # terms
	    tmp1 = psi(a_features[k]) - psi(a_features[k] + b_features[k]) - \
	        u_features[k]/v_features[k]

	    tmp2 = 0.0
	    for i in xrange(1, M+1):
	        tmp2 -= (1./i)*gamma_ratios_(a_features[k] + b_features[k],
		    a_features[k], i)*(v_features[k]/(v_features[k] + i))**\
		    u_features[k]

	    lb_4_features += \
		(1 - vphi_features[0, k])*(sum(phi_features[:, k])*tmp1 + \
					   sum(1 - phi_features[:, k])*tmp2)

            # Update 'lb_8_features'. This is the last term in equation 10 of
	    # ref [1] (in my notes, page 11.8.13.4).
	    lb_8_features += (1-vphi_features[0, k])*(gammaln(u_features[k]) - \
		    (u_features[k] - 1)*psi(u_features[k]) - \
		    log(v_features[k]) + u_features[k])	    

    lb_2_instances = 0.0
    lb_5_instances = 0.0
    lb_7_2 = 0.0
    if vphi_instances is not None and phi_instances is not None and \
	a_instances is not None and b_instances is not None and \
	alpha_instances is not None:
	K = vphi_instances.shape[1]
	for k in xrange(0, K):
	    lb_2_instances += vphi_instances[0, k]*( \
		sum(phi_instances[:, k])*(psi(a_instances[k]) - \
		      psi(a_instances[k] + b_instances[k])) + \
   	        sum(1 - phi_instances[:, k])*(psi(b_instances[k]) - \
		      psi(a_instances[k] + b_instances[k])))

	    lb_5_instances += log(alpha_instances) + (alpha_instances - 1)*\
		(psi(b_instances[k]) - psi(a_instances[k] + b_instances[k]))

            lb_7_2 -= log((1./beta(a_instances[k], \
              b_instances[k]).clip(1e-300)).clip(max=1e300)) + \
              (a_instances[k] - 1)*(psi(a_instances[k]) - \
              psi(a_instances[k] + b_instances[k])) + \
              (b_instances[k] - 1)*(psi(b_instances[k]) - \
              psi(a_instances[k] + b_instances[k]))

    lb_2_features = 0.0    
    lb_5_features = 0.0
    lb_7_5 = 0.0
    if vphi_features is not None and phi_features is not None and \
	a_features is not None and b_features is not None and \
	alpha_features is not None:
	K = vphi_features.shape[1]
	for k in xrange(0, K):
	    lb_2_features += vphi_features[0, k]*( \
		sum(phi_features[:, k])*(psi(a_features[k]) - \
		      psi(a_features[k] + b_features[k])) + \
   	        (1 - sum(phi_features[:, k]))*(psi(b_features[k]) - \
		      psi(a_features[k] + b_features[k])))

	    lb_5_features += log(alpha_features) + (alpha_features - 1)*\
		(psi(b_features[k]) - psi(a_features[k] + b_features[k]))

            lb_7_5 -= log((1./beta(a_features[k], \
              b_features[k]).clip(1e-300)).clip(max=1e300)) + \
              (a_features[k] - 1)*(psi(a_features[k]) - \
              psi(a_features[k] + b_features[k])) + \
              (b_features[k] - 1)*(psi(b_features[k]) - \
              psi(a_features[k] + b_features[k]))    

    lb_6_instances = 0.0
    lb_7_1 = 0.0
    if vphi_instances is not None:
        K = vphi_instances.shape[1]
        R = vphi_instances.shape[0]
        
	lb_7_1 += sum(vphi_instances*log(vphi_instances.clip(1e-300)))

        for r in xrange(0, R):
	    lb_6_instances -= 0.9375*(sum(np.outer(vphi_instances[r, :], \
		vphi_instances[r, :])) - np.dot(vphi_instances[r, :], \
			vphi_instances[r, :]) + sum(vphi_instances[r, :]))

    lb_6_features = 0.0
    lb_7_4 = 0.0
    if vphi_features is not None:
        K = vphi_features.shape[1]
        R = vphi_features.shape[0]
        
	lb_7_4 += sum(vphi_features*log(vphi_features.clip(1e-300)))

        for r in xrange(0, R):
	    lb_6_features -= 0.9375*(sum(np.outer(vphi_features[r, :], \
		vphi_features[r, :])) - np.dot(vphi_features[r, :], \
			vphi_features[r, :]) + sum(vphi_features[r, :]))    

    lb_7_3 = 0.0
    if phi_instances is not None:
        lb_7_3 -= sum(phi_instances*log(phi_instances.clip(1e-300)) + \
		(1 - phi_instances)*log((1 - phi_instances).clip(1e-300)))

    lb_7_6 = 0.0
    if phi_features is not None:
        lb_7_6 -= sum(phi_features*log(phi_features.clip(1e-300)) - \
		(1 - phi_features)*log((1 - phi_features).clip(1e-300)))
	
    lower_bound = lb_1_1 + lb_7_7 + lb_1_2 + lb_3_instances + lb_3_features + \
	lb_4_instances + lb_4_features + lb_8_instances + lb_8_features + \
	lb_2_instances + lb_5_instances + lb_7_2 + lb_2_features + \
        lb_5_features + lb_7_5 + lb_6_instances + lb_6_features + lb_7_1 + \
	lb_7_4 + lb_7_3 + lb_7_6

    return lower_bound

def gamma_ratios_(a, b, n):
    """Computes (gamma(a)/gamma(a+n))*(gamma(b+n)/gamma(b)) using the gamma
    function recursion property.

    Parameters
    ----------
    a : float
        Real number greater than 0

    b : float
        Real number greater than 0

    n : int
        Non-negative integer

    Returns
    -------
    r : float
        The value (gamma(a)/gamma(a+n))*(gamma(b+n)/gamma(b))
    """
    assert a > 0, "Parameter a must be greater than 0"
    assert b > 0, "Parameter b must be greater than 0"
    assert n >= 0, "Parameter n must be greater than or equal to zero"

    r = np.prod((b + linspace(0, n-1, n))/(a + linspace(0, n-1, n)))

    return r

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

def phi_instance_obj_(phi, phi_instances, phi_features, mu, cov, sig_betas,
                      t1, t2, t3, t4, n, k):
    """Objective function for updating phi_instances for a given instance and
    latent feature.
    
    Parameters
    ----------
    phi : float
        Must be in the interval (0, 1). This is the location at which the
	objective function will be evaluated.
    
    phi_instances : array, shape ( N, K ), 
        Matrix of Bernoulli parameter values relating to the association
        between instances and latent features.

    phi_features : array, shape ( D, K )
        Matrix of Bernoulli parameter values relating to the association
        between observed features and latent features.

    mu : array, shape ( N, K, D )
        The posterior means of the GP regressors. 'N' is the number of
        instances, 'K' is the number of latent features in the truncated beta
        process, and 'D' is the number of observed features.    

    cov : array, shape ( N, N, K, D )
        Collection of covariance matrices for the posterior GP regressors. 'N'
	is the number of instances, 'K' is the number of latent features in the
	truncated beta process, and 'D' is the number of observed features.
	Note that the covariance matrix is only computed for those instances
	that -- for a given observed feature and latent feaure ('d' and 'k') --
	have a probability of belonging to that GP that is greater than
        'prob_thresh'. The other elements of the covariance matrix are set to
	NaN.

    sig_betas : array, shape ( D )
        Precision values for each of the 'D' observed features.

    t1 : array, shape ( N, D )
        Array of floats needed for computing the objective function value.

    t2 : array, shape ( N, K, D )
        Array of floats needed for computing the objective function value.

    t3 : float
        Needed for computing the objective function value.

    t4 : float
        Needed for computing the objective function value.

    n : integer
        The phi_instances instance for which to compute the objective function.

    k : integer
        The phi_instances latent feature for which to compute the objective
	function.

    Notes
    -----
    See written notes on 2.2.14.1 for more about inputs 't1', 't2', 't3' and
    't4'. These are precomputed quantities needed to compute the objective
    function value.
    """
    D = phi_features.shape[0]
    K = phi_instances.shape[1]

    val = phi*(t3 + t4) - (phi*log(phi) + (1 - phi)*log(1 - phi))

    phi_instances_vec = np.copy(phi_instances[n, :])
    phi_instances_vec[k] = phi
    for d in xrange(0, D):
        ep = 1 - prod(1 - phi_instances_vec[:]*phi_features[d, :])
        ids = np.logical_not(np.isnan(mu[n, :, d]))
        val += ep*(sum(phi_instances_vec[:]*phi_features[d, :]*t2[n, :, d]) + \
                   t1[n, d] - 0.5*sig_betas[d]*\
                   expec_zf_sqr(phi_instances_vec[ids], phi_features[d, ids], \
                                mu[n, ids, d], cov[n, n, ids, d]))

    # We return the negative because we will be invoking a minimization routine
    return -val

def phi_feature_obj_(phi, phi_instances, phi_features, mu, cov, sig_betas,
                     t1, t2, t3, t4, d, k):
    """Objective function for updating phi_features for a observed and latent
    feature.
    
    Parameters
    ----------
    phi : float
        Must be in the interval (0, 1). This is the location at which the
	objective function will be evaluated.
    
    phi_instances : array, shape ( N, K ), 
        Matrix of Bernoulli parameter values relating to the association
        between instances and latent features.

    phi_features : array, shape ( D, K )
        Matrix of Bernoulli parameter values relating to the association
        between observed features and latent features.

    mu : array, shape ( N, K, D )
        The posterior means of the GP regressors. 'N' is the number of
        instances, 'K' is the number of latent features in the truncated beta
        process, and 'D' is the number of observed features.    

    cov : array, shape ( N, N, K, D )
        Collection of covariance matrices for the posterior GP regressors. 'N'
	is the number of instances, 'K' is the number of latent features in the
	truncated beta process, and 'D' is the number of observed features.
	Note that the covariance matrix is only computed for those instances
	that -- for a given observed feature and latent feaure ('d' and 'k') --
	have a probability of belonging to that GP that is greater than
        'prob_thresh'. The other elements of the covariance matrix are set to
	NaN.

    sig_betas : array, shape ( D )
        Precision values for each of the 'D' observed features.

    t1 : array, shape ( N, D )
        Array of floats needed for computing the objective function value.

    t2 : array, shape ( N, K, D )
        Array of floats needed for computing the objective function value.

    t3 : float
        Needed for computing the objective function value.

    t4 : float
        Needed for computing the objective function value.

    d : integer
        The phi_features observed feature for which to compute the objective
	function value.

    k : integer
        The phi_features latent feature for which to compute the objective
	function value.

    Notes
    -----
    See written notes on 2.2.14.1 and 2.3.14.1 for more about inputs 't1',
    't2', 't3' and 't4'. These are precomputed quantities needed to compute the
    objective function value.
    """
    N = phi_instances.shape[0]
    K = phi_instances.shape[1]

    val = phi*(t3 + t4) - (phi*log(phi) + (1 - phi)*log(1 - phi))

    phi_features_vec = np.copy(phi_features[d, :])
    phi_features_vec[k] = phi
    for n in xrange(0, N):
        ep = 1 - prod(1 - phi_features_vec[:]*phi_instances[n, :])
        ids = np.logical_not(np.isnan(mu[n, :, d]))
        val += ep*(sum(phi_features_vec[:]*phi_instances[n, :]*t2[n, :, d]) + \
                   t1[n, d] - 0.5*sig_betas[d]*\
                   expec_zf_sqr(phi_instances[n, ids], phi_features_vec[ids], \
                                mu[n, ids, d], cov[n, n, ids, d]))

    # We return the negative because we will be invoking a minimization routine
    return -val

def expec_zf_sqr(phi_instance_vec, phi_feature_vec, mu_vec, cov_vec):
    """Compute the expected value of (\sum_{k=1}^{K}(z_{k}F_{n,d}^{(k)})^2,
    where z_{k} is the product of latent variables z_{n,k}*z_{d,k}, elements
    of the indicator matrices associating instances to latent features and
    observed features to latent variables, respectively. F_{n,d}^{(k)}
    corresponds to the instance n of the k^th Gaussian process for observed
    feature k.

    Parameters
    ----------
    phi_instance_vec : array, shape ( K )
        Vector of Bernoulli parameter values relating to the association
        between a specific instance and the latent features. 'K' is the number
        of latent features in the truncated beta process.

    phi_feature_vec : array, shape ( K )
        Vector of Bernoulli parameter values relating to the association
        between a specific observed feature and the latent features. 'K' is
        the number of latent features in the truncated beta process.

    mu_vec : array, shape ( K )
        Vector of GP posterior means of for specific instance and observed
        feature across all latent features. 'K' is the number of latent
        features in the truncated beta process.

    cov_vec : array, shape ( K )
        Elements of the GP posterior covariance matrices for specific
        instance and observed feature across all latent features, 'K'.

    Returns
    -------
    expec : float
        The expectation 
    """
    mat = outer(phi_instance_vec[:]*phi_feature_vec[:], \
                phi_instance_vec[:]*phi_feature_vec[:])*\
          outer(mu_vec[:], mu_vec[:])

    expec = sum(mat)

    # Now replace the diagonal elements
    expec += sum(phi_instance_vec[:]*phi_feature_vec[:]*\
                 (mu_vec[:]**2 + cov_vec[:])) - trace(mat)

    return expec
