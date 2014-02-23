import pdb

import numpy as np

def compute_kernel_func_vals(data1, data2, kernel):
    """Computes the kernel function value between the two input vectors using
    the specified kernel.

    Parameters
    ----------
    data1 : array, shape ( n_samples, dimension )
        'n_samples' is the number of data points and 'dimension' is the
        dimension.

    data2 : array, shape ( n_samples, dimension )
        'n_samples' is the number of data points and 'dimension' is the
        dimension

    kernel : list
        A list with two elements. The first element is a string indicating the
        name of the kernel (see the 'Notes' below for currently supported
        kernels). The second element is an array of length 'params', where
        'params' are the scalar parameter values controlling the behavior of
        the kernel

    Returns
    -------
    vals : array, shape ( n_samples, 1 )
        The scalar kernel function values

    Notes
    -----
    Currently supported kernels:
    Gaussian kernel: kernel[0] = 'gaussian'. The equation of the kernel is
    $\theta_{1}^{2}\exp\{-\frac{\theta_{2}}{2}||x_{n}-x_{m}||^{2}\}$. The
    parameters for this kernel are: kernel[1] = array([$\theta_{1}$,
    $\theta_{2}$])

    Constant kernel: kernel[0] = 'constant'. All kernel values are set to
    the value specified by kernel[1]

    Gaussian kernel: kernel[0] = '{heterogenous\_gaussian}'. The equation of 
    the kernel is$\theta_{1}^{2}\exp\{-\frac{1}{2}\sum_{d=1}^D w_d (x_i^{(d)} 
    - x_j^{(d)})^{2}\}$. The parameters for this kernel are: kernel[1] = array
    ([$\theta_{1}$, $w_1, \ldots , w_D$])

    Linear kernel: kernel[0] = 'linear'. The equation of the kernel is
    $\theta_{0}+\theta_{1}\sum_{d=1}^{D}(x_{d}-\theta_{2})(x_{d}^{\prime}
    -\theta_{2})$. There are three parameters for this kernel: kernel[1] =
    array([$\theta_{0}$, $\theta_{1}$, $\theta_{2}$])

    Periodic kernel: kernel[0] = 'periodic'. The equation of the kernel is
    $\theta_{0}\exp(-\frac{1}{\theta_{1}}\sum_{d=1}^{D}\sin^2
    (\frac{\pi}{\theta_{2}}(x_{d}-x^{\prime}_{d})))$. There are three
    parameters for this kernel: kernel[1] = array([$\theta_{0}$, $\theta_{1}$,
    $\theta_{2}$])    
    """
    if data1.shape[0] != data2.shape[0]:
        raise ValueError("Number of samples must be same")

    assert kernel[0] in ['gaussian', 'heterogenous_gaussian', 'constant', \
                         'linear', 'periodic'], "Unsupported kernel"

    n_samples = data1.shape[0]
    n_dim = data1.shape[1]
    params = kernel[1]
    vals = np.zeros((n_samples,1))

    # Gaussian kernel case
    if kernel[0] == 'gaussian':
        diff = data1 - data2
        tmp  = (diff*diff).sum(axis = 1)
        vals = (params[0]**2)*np.exp(-params[1]*tmp/2.0)

    # Constant kernel case
    elif kernel[0] == 'heterogenous_gaussian':
        assert params.shape[0] == n_dim+1, "Incorrect number of parameters. \
        heterogenous Gaussian kernel requires dim+1"
        
        w = params[1:n_dim+1]
        diff = np.matrix((data1-data2)**2) 
        tmp = w * diff.T
        vals = (params[0] ** 2 * np.exp (-tmp/2.)).A

    # Linear kernel case
    elif kernel[0] == 'linear':
        assert params.shape[0] == 3, "Incorrect number of parameters. \
        Linear kernel requires three"

        vals = params[0] + \
          params[1]*np.sum((data1-params[2])*(data2-params[2]), 1)
        
    # Periodic kernel case
    elif kernel[0] == 'periodic':
        assert params.shape[0] == 3, "Incorrect number of parameters. \
        Periodic kernel requires three"
        assert params[1] != 0., "Second paramter cannot be zero for the \
        periodic kernel"
        assert params[2] != 0., "Third paramter cannot be zero for the \
        periodic kernel"

        vals = params[0]*np.exp(-(1/params[1])*\
          np.sum(np.sin((np.pi/params[2])*(data1 - data2))**2, 1))

    # Constant case
    elif kernel[0] == 'constant':
        vals = params[0]
    else:
        vals = None

    return vals
