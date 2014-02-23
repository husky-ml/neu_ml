import pdb

import numpy as np
from numpy import pi
from neu_ml.code.utils.compute_kernel_func_vals \
    import compute_kernel_func_vals

def compute_gram_mat(data, kernel):
    """Compute the Gram matrix corresponding to input data using the specified
    kernel

    Parameters
    ----------
    data : array, shape ( n_instances, dimension ) or shape ( n_instances )
        The input data over which to compute the Gram matrix

    kernel : list
        A list with two elements. The first element is a string indicating the
        type of kernel. The second element is an array indicating the
        parameter values of the kernel

    Returns
    -------
    gram_mat : array, shape ( n_instances, n_instances )
        The Gram matrix

    Notes
    -----
    Currently supported kernels:
    Gaussian kernel: kernel[0] = 'gaussian'. The equation of the kernel is
    $\theta_{1}^{2}\exp\{-\frac{\theta_{2}}{2}||x_{n}-x_{m}||^{2}\}$. The
    parameters for this kernel are: kernel[1] = array([$\theta_{1}$,
    $\theta_{2}$])

    Linear kernel: kernel[0] = 'linear'. The equation of the kernel is
    $\theta_{0}+\theta_{1}\sum_{d=1}^{D}(x_{d}-\theta_{2})(x_{d}^{\prime}
    -\theta_{2})$. There are three parameters for this kernel: kernel[1] =
    array([$\theta_{0}$, $\theta_{1}$, $\theta_{2}$])

    Periodic kernel: kernel[0] = 'periodic'. The equation of the kernel is
    $\theta_{0}\exp(-\frac{1}{\theta_{1}}\sum_{d=1}^{D}\sin^2
    (\frac{\pi}{\theta_{2}}(x_{d}-x^{\prime}_{d})))$. There are three
    parameters for this kernel: kernel[1] = array([$\theta_{0}$, $\theta_{1}$,
    $\theta_{2}$])
    
    Gaussian kernel: kernel[0] = '{heterogenous\_gaussian}'. The equation of 
    the kernel is$\theta_{1}^{2}\exp\{-\frac{1}{2}\sum_{d=1}^D w_d (x_i^{(d)} 
    - x_j^{(d)})^{2}\}$. The parameters for this kernel are: kernel[1] = array
    ([$\theta_{1}$, $w_1, \ldots ,     
    """
    n_instances = data.shape[0]
    if len(data.shape) == 1:
        dim = 1
        data = np.atleast_2d(data).T
    else:
        dim = data.shape[1]
        
    params = kernel[1]

    assert kernel[0] in ['gaussian', 'linear', 'periodic', 
        'heterogenous_gaussian'], "Unsupported kernel"

    if kernel[0] == 'gaussian':
        assert params.shape[0] == 2, "Incorrect number of parameters. \
        Gaussian kernel requires two"
        
        accum_mat = np.zeros([n_instances, n_instances])
        for d in xrange(0, dim):
            tmp_mat1 = np.array([data[:, d]]*n_instances)
            tmp_mat2 = np.array([data[:, d]]*n_instances).T            
            accum_mat += (tmp_mat1 - tmp_mat2)**2

        gram_mat = (params[0]**2)*np.exp(-0.5*params[1]*accum_mat)
    elif kernel[0] == 'linear':
        assert params.shape[0] == 3, "Incorrect number of parameters. \
        Linear kernel requires three"
        
        accum_mat = np.zeros([n_instances, n_instances])
        ones_mat = np.ones([n_instances, n_instances])
        for d in xrange(0, dim):
            tmp_mat1 = np.array([data[:, d]]*n_instances)
            tmp_mat2 = np.array([data[:, d]]*n_instances).T
            accum_mat = (tmp_mat1 - params[2]*ones_mat)*\
                (tmp_mat2 - params[2]*ones_mat)

        gram_mat = ones_mat*params[0] + params[1]*accum_mat
    elif kernel[0] == 'periodic':
        assert params.shape[0] == 3, "Incorrect number of parameters. \
        Periodic kernel requires three"
        assert params[1] != 0., "Second paramter cannot be zero for the \
        periodic kernel"
        assert params[2] != 0., "Third paramter cannot be zero for the \
        periodic kernel"
        
        accum_mat = np.zeros([n_instances, n_instances])
        for d in xrange(0, dim):
            tmp_mat1 = np.array([data[:, d]]*n_instances)
            tmp_mat2 = np.array([data[:, d]]*n_instances).T
            accum_mat = np.sin((pi/params[2])*(tmp_mat1-tmp_mat2))**2

        gram_mat = params[0]*np.exp((-1./params[1])*accum_mat)
    elif kernel[0] == 'heterogenous_gaussian':
        assert params.shape[0] == dim+1, "Incorrect number of parameters. \
        heterogenous Gaussian kernel requires dim+1"
        
        accum_mat = np.zeros([n_instances, n_instances])
        
        w = params[1:dim+1]
        
        for d in xrange(0, dim):
            tmp_mat1 = np.array([data[:, d]]*n_instances)
            tmp_mat2 = tmp_mat1.T            
            accum_mat += w[d] * (tmp_mat1 - tmp_mat2)**2

        gram_mat = (params[0]**2)*np.exp(-0.5*accum_mat)
        

    return gram_mat    
