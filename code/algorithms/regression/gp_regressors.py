import pdb

import numpy as np
from scipy.linalg import pinv2
from neu_ml.code.utils.compute_gram_mat \
    import compute_gram_mat
from neu_ml.code.utils.compute_kernel_func_vals import \
    compute_kernel_func_vals

class GPRegressor:
    """Class coordinating Gaussian Process regression

    Parameters
    ----------
    kernel : list
        A list with two elements. The first element is a string indicating the
        type of kernel. The second element is an array indicating the parameter
        values of the kernel. The kernel is used to compute the Gram matrix,
        and the "augmented" Gram matrix (containing entries not only for inputs
        corresponding to observed data, but also for domain locations at which
        you want to predict target values) which is needed for regression.

    inputs : array, shape ( n_instances, input_dim ), optional
        Array of inputs (aka domain locations) of observed data

    targets : array, shape ( n_instances, target_dim ), optional
        Array of observed target values at corresponding 'input' locations

    gram_mat : array, shape ( n_instances, n_instances ), optional
        Gram matrix corresponding to observed input data. If the Gram matrix is
        not specified using this method, or if the currently stored Gram matrix
        does not have the expected shape, the Gram matrix will be computed
        using 'kernel'

    prob_mat : array, shape ( n_instances, n_instances ), optional
        Diagonal array. Each diagonal element indicates the probability that
        the corresponding target belongs to the Gaussian Process. For
        typical Gaussian Process regression, this matrix should simply be the
        identity matrix

    precision : double, optional
        The known noise precision on the target values

    See also
    --------
    Refer to 'compute_kernel_func_vals.py' for supported kernel types
    """
    
    def __init__(self, kernel, inputs=None, targets=None, gram_mat=None,
                 prob_mat=None, precision=None):
        self._kernel = kernel

        # Internal machinery needs both the inputs and the targets to be 2D
        # matrices. Cast to 2D matrices if necessary.
        if inputs.ndim == 1:
            self._inputs = np.atleast_2d(inputs).T
        else:
            self._inputs = inputs

        if targets.ndim == 1:
            self._targets = np.atleast_2d(targets).T
        else:
            self._targets = targets
            
        self._gram_mat = gram_mat
        self._prob_mat = prob_mat
        self._precision = precision

        # The class supports subsampling the Gram and probability matrices, and
        # the input-target pairs. This is useful for efficient implementation
        # of some algorithms. Subsampling involves specifying
        # '_active_obs_indices', which is a subset of the observation index set.
        # These are the indices of the input-target pairs that are likely to
        # belong to Gaussian Process. If not specified, all input-target pairs
        # will be used.
        self._active_obs_indices = None

        # The covariance and inverse covariance matrices will need to be
        # computed each time the Gram matrix, probability matrix, active
        # observation indices, or precision are changed. These variables are
        # therefore meant to be private and should only be accessed through
        # the 'get_active_cov_mat' and 'get_inv_active_cov_mat' functions
        self._active_cov_mat = None
        self._inv_active_cov_mat = None

        self._inv_gram_mat = None

        # Record the number of inputs and targets appropriately, and ensure
        # that they are equal
        self.n_inputs = None
        self.n_targets = None

        if inputs is not None:        
            self.n_inputs = inputs.shape[0]

        if targets is not None:    
            self.n_targets = targets.shape[0]

        if inputs is not None and targets is not None:
            if self.n_inputs != self.n_targets:
                raise ValueError("Number of inputs not equal to number of \
                                targets")

        # Typical Gaussian process regression assumes that each of the target
        # values has 100% probability of belonging to the process of interest.
        # This corresponds to an identity matrix for 'prob_mat'. Make this the
        # default.
        if prob_mat is None and self.n_inputs is not None:
            self._prob_mat = np.eye(self.n_inputs)

    def set_gram_mat(self, gram_mat):
        """Set the (pre-computed) Gram matrix corresponding to the observed
        input data

        Parameters
        ----------
        gram_mat :  array, shape ( n_instances, n_instances )
            Gram matrix corresponding to the input data. 
        """        
        self._gram_mat = gram_mat

        n_rows = self._gram_mat.shape[0]
        n_cols = self._gram_mat.shape[1]

        if n_rows != n_cols:
            raise ValueError("Gram matrix is not square")

    def get_gram_mat(self):
        """Get the Gram matrix; compute if necessary.

        Returns
        -------
        gram_mat :  array, shape ( n_instances, n_instances )
            Gram matrix corresponding to the input data. 
        """
        if self._gram_mat is None:
            self.compute_gram_mat()

        return self._gram_mat

    def get_active_gram_mat(self):
        """Get the active Gram matrix

        The active Gram matrix is defined by the rows and columns corresponding
        to the currently active observations, so it will in general be
        smaller than the complete Gram matrix.

        Returns
        -------
        active_gram_mat : array, shape ( n_instances, n_instances )
            Gram matrix corresponding to the input data. 
        """
        # Get the currently active observation indices
        active_obs_indices = self.get_active_obs_indices()
        n_active = active_obs_indices.shape[0]

        assert n_active > 0, "Number of active instances is not greater than 0"
        
        if self._gram_mat is None:
            self.compute_gram_mat()

        active_gram_mat = \
            self.sub_sample_mat(self._gram_mat, active_obs_indices)

        return active_gram_mat

    def get_inv_active_gram_mat(self):
        """Get the inverse of the active Gram matrix.

        The active Gram matrix is defined by the rows and columns corresponding
        to the currently active observations, so the inverse of the active Gram
        matrix will in general be smaller than the inverse of the complete Gram
        matrix.

        Returns
        -------
        inv_active_gram_mat : array, shape ( n_instances, n_instances )
            Inverse of the active Gram matrix.
        """
        # Get the currently active observation indices
        active_obs_indices = self.get_active_obs_indices()
        n_active = active_obs_indices.shape[0]

        assert n_active > 0, "Number of active instances is not greater than 0"
        
        if self._gram_mat is None:
            self.compute_gram_mat()

        active_gram_mat = \
            self.sub_sample_mat(self._gram_mat, active_obs_indices)

        # Use SVD to compute the inverse. This is done to protect against
        # numerical instability
        [U, S, Vh] = np.linalg.svd(active_gram_mat)

        inv_S = np.zeros((active_gram_mat.shape[0], active_gram_mat.shape[0]))
        for i in np.arange(0, active_gram_mat.shape[0]):
            if S[i] > np.spacing(1):
                inv_S[i, i] = 1.0/S[i]

        inv_active_gram_mat = np.dot(Vh.T, np.dot(inv_S, U.T))
 
        return inv_active_gram_mat    

    def set_input_target_pairs(self, inputs, targets):
        """Set observed input-target pairs.

        Parameters
        ----------
        inputs : array, shape ( n_instances, input_dim )
            Array of inputs (aka domain locations) of observed data

        targets : array, shape ( n_instances, 1 )
            Array of observed target values at corresponding 'input' locations
        """
        self._inputs  = inputs
        self._targets = targets

        self.n_inputs = inputs.shape[0]
        self.n_targets = targets.shape[0]

        # The number of inputs must be the same as the number of targets
        if self.n_inputs != self.n_targets:
            raise ValueError("Number of inputs not equal to number of targets")

    def set_kernel(self, kernel):
        """Set the Gaussian process kernel.

        Calling this function will trigger recomputation of the internal Gram
        matrix if 'inputs' is not empty

        Parameters
        ----------
        kernel : list
            A list with two elements. The first element is a string indicating
            the type of kernel. The second element is an array indicating the
            parameter values of the kernel.
        """        
        self._kernel = kernel

        if self._inputs is not None:
            self.compute_gram_mat()

    def set_target_probabilities(self, prob_mat):
        """Set the probability matrix, which indicates, for each instance, the
        probability of belonging to the Gaussian Process

        Parameters
        ----------
        prob_mat : array, shape ( n_instances, n_instances ), optional
            Diagonal array. Each diagonal element indicates the probability
            that the corresponding target belongs to the Gaussian Process. For
            typical Gaussian Process regression, this matrix should simply be
            the identity matrix
        """
        self._prob_mat = prob_mat

        # Matrix must be square
        n_rows = self._prob_mat.shape[0]
        n_cols = self._prob_mat.shape[1]
        if n_rows != n_cols:
            raise ValueError("Probability matrix is not square")

    def get_inputs(self):
        """Get the input values.

        Returns
        -------
        inputs : array, shape ( n_instances, input_dim )
            Array of inputs (aka domain locations) of observed data        
        """
        assert self._inputs is not None, "Attempt to get inputs, but inputs\
        are None"

        return self._inputs

    def get_targets(self):
        """Get the target values.

        Returns
        -------
        targets : array, shape ( n_instances, target_dim )
            Array of observed target values at corresponding 'input' locations
        """
        assert self._targets is not None, "Attempt to get targets, but targets\
        are None"

        return self._targets

    def set_precision(self, precision):
        """Set the precision (inverse variance) for the target variables.

        Paramters
        ---------
        precision : float
            Precision (inverse variance) for the target variables
        """
        self._precision = precision

        # Precision must be greater than 0.0
        if self._precision <= 0.0:
            raise ValueError("Precision <= 0.0")

    def set_active_obs_indices(self, indices):
        """Indicate the observation indices to be apply when subsampling the
        Gram matrix and the probability matrix.

        Parameters
        ----------
        indices : array, shape ( n )
            Array of valid indices corresponding to input observations
        """
        if np.min(indices) < 0:
            raise ValueError("Actice indices can not be negative")
        
        self._active_obs_indices = indices

    def get_target_predictions_at_inputs(self):
        """Computes the mean and variance at each of the specified (observed)
        input locations using the specified Gaussian Process information.

        This is useful when the 'active_obs_indices' indicates a
        subset of observations to use when making predictions and when we want
        to make those predictions at each of the observed input locations.

        Returns
        -------
        means : array, shape ( n_instances, target_dim )
            The array of mean values for each target dimension across at each
            instance input location.

        vars : array, shape ( n_instances, target_dim )
            The array of variances for each target dimension across each
            instance input location.
        """
        if self._targets.ndim == 1:
            target_dim = 1
        else:
            target_dim = self._targets.shape[1]

        # Pre-allocate space for the outputs
        means = np.zeros([self._inputs.shape[0], target_dim])
        vars  = np.zeros([self._inputs.shape[0], target_dim])

        # Get the currently active observation indices
        active_obs_indices = self.get_active_obs_indices()
        n_active = active_obs_indices.shape[0]

        # If the Gram matrix is 'None', compute it now
        if self._gram_mat is None:
            self.compute_gram_mat()
        
        # Sub-sample the targets.
        targets = np.zeros((n_active, target_dim))
        for i in np.arange(0, n_active):
            targets[i, :] = self._targets[active_obs_indices[i], :]

        # Get the covariance matrix and the inverse covariance matrix. Note
        # that in general the size of the matrices can be smaller than the
        # number of data instances given that only some instances may be
        # "active"
        cov_mat = self.get_active_cov_mat()
        inv_cov_mat = self.get_active_inv_cov_mat()

        # Now loop over each of the domain points requested to have means and
        # variances computed for.
        for d in np.arange(0, target_dim):
            for i in np.arange(0, self._inputs.shape[0]):
                # Compute the vector 'k'. See Eq. 6.65 and text in Bishop2006
                # for a description. 
                k = self.extract_vec(self._gram_mat, i, active_obs_indices)
    
                # Compute the scalar 'c'. See Eq. 6.65 in Bishop2006 and text
                # for more details
                c = self._gram_mat[i, i] + 1.0/self._precision

                # Now compute the means and variances according to the
                # equations in Bishop's 'Machine Learning and Pattern
                # Recognition', equations 6.66 and 6.67
                means[i, d] = np.dot(k, np.dot(inv_cov_mat, targets[:, d]))
                vars[i, d]  = c - np.dot(k, np.dot(inv_cov_mat, k)) 

        return means, vars

    def get_target_predictions(self, pts):
        """Computes the mean and variance at a set of locations designated in
        'pts' using the specified Gaussian Process information.

        Parameters
        ----------
        pts : array, shape ( n_points, input_dim )
            Matrix of query points, where 'n_points' is the number of query points,
            and 'input_dim' is the dimension of the input variables

        Returns
        -------
        means : array, shape ( n_points )
            Mean values computed relative to the Gaussian Process regression
            information. Here 'n_points' is the number of query points; these
            values correspond to the query points in 'points'

        vars : array, shape ( n_points )
            Variance values computed relative to the Gaussian Process
            regression information. Here 'n_points' is the number of query
            points; these values correspond to the query points in 'points'

        References
        ----------
        Bishop2006 (see bibFile.bib) provides the equations used here. In
        particular, see Eq. 6.66 and Eq. 6.67 in section 'Gaussian Processes for
        Regression'
        """
        n_pts = pts.shape[0]

        # Input points are expected to be represented as a 2D matrix. Cast if
        # necessary
        if pts.ndim == 1:
            _pts = np.atleast_2d(pts).T
        else:
            _pts = pts

        if self._inputs.ndim == 1:
            input_dim = 1
        else:
            input_dim = self._inputs.shape[1]

        if self._targets.ndim == 1:
            target_dim = 1
        else:
            target_dim = self._targets.shape[1]

        # Pre-allocate space for the outputs
        means = np.zeros([n_pts, target_dim])
        vars  = np.zeros([n_pts, target_dim])

        # If the Gram matrix is 'None', compute it now
        if self._gram_mat is None:
            self.compute_gram_mat()

        # Get the currently active set of indices
        active_obs_indices = self.get_active_obs_indices()
        n_active = active_obs_indices.shape[0]

        # Get the currently active inputs and targets
        targets = self.get_active_targets()
        inputs = self.get_active_inputs()

        # Get the covariance matrix and the inverse covariance matrix
        cov_mat = self.get_active_cov_mat()
        inv_cov_mat = self.get_active_inv_cov_mat()

        # Now loop over each of the domain points requested to have means and
        # variances computed for.
        for d in np.arange(0, target_dim):
            for i in np.arange(0, n_pts):
                # Compute the vector 'k'. See Eq. 6.65 and text in Bishop2006
                # for a description
                k = np.zeros(n_active)

                for j in np.arange(0, n_active):
                    k[j] = compute_kernel_func_vals(np.atleast_2d(_pts[i,:]).T,
                                    np.atleast_2d(inputs[j,:]).T, self._kernel)
    
                # Compute the scalar 'c'. See Eq. 6.65 in Bishop2006 and text
                # for more details
                c = compute_kernel_func_vals(np.atleast_2d(_pts[i,:]).T,
                        np.atleast_2d(_pts[i,:]).T, self._kernel) + \
                        1.0/self._precision

                means[i, d] = \
                    np.dot(k, np.dot(inv_cov_mat, targets[:, d])) # Eq. 6.66
                vars[i, d]  = c - np.dot(k, np.dot(inv_cov_mat, k)) # Eq. 6.67

        return means, vars

    def get_active_inputs(self):
        """Get the currently active input data.

        At any given time, only a subset of the input-target pairs may be
        active (i.e. used for defining the Gaussian Process). Thie method
        provides access to the inputs that are currently active. If no set of
        active indices has been provided, all inputs will be assumed active. If
        no inputs have been specified at the time of method execution, this
        method will return 'None'.

        Returns
        -------
        active_inputs : array, shape ( N, input_dim )
            The currently active inputs, where 'N' is the number of currently
            active indices.
        """
        if self._inputs is None:
            return None
        
        active_obs_indices = self.get_active_obs_indices()
        n_active = active_obs_indices.shape[0]

        assert np.max(active_obs_indices) < self._inputs.shape[0], "Active \
        observation indices can not exceed input extent"

        if self._inputs.ndim == 1:
            input_dim = 1
        else:
            input_dim = self._inputs.shape[1]
            
        active_inputs = np.zeros((n_active, input_dim))
        for i in np.arange(0, n_active):
            active_inputs[i, :] = self._inputs[active_obs_indices[i]]

        return active_inputs

    def get_active_targets(self):
        """Get the currently active target data.

        At any given time, only a subset of the input-target pairs may be
        active (i.e. used for defining the Gaussian Process). Thie method
        provides access to the targets that are currently active. If no set of
        active indices has been provided, all targets will be assumed active.
        If no targets have been specified at the time of method execution, this
        method will return 'None'.

        Returns
        -------
        active_targets : array, shape ( N, input_dim )
            The currently active targets, where 'N' is the number of currently
            active indices.
        """
        if self._targets is None:
            return None
        
        active_obs_indices = self.get_active_obs_indices()
        n_active = active_obs_indices.shape[0]

        assert np.max(active_obs_indices) < self._targets.shape[0], "Active \
        observation indices can not exceed input extent"

        target_dim = self._targets.shape[1]
        active_targets = np.zeros((n_active, target_dim))
        for i in np.arange(0, n_active):
            active_targets[i, :] = self._targets[active_obs_indices[i]]

        return active_targets

    def get_active_obs_indices(self):
        """Get the currently active observation indices.

        At any given time, only a subset of the input-target pairs may be
        "active" (i.e. used for defining the Gaussian Process). Thie method
        provides access to the set of indices corresponding to the input-
        target pairs that are currently active. Note that the user should use
        this method to access the set of active indices and should not
        access them directly. In other words, don't use
        "my_instances._active_obs_indices", as "_active_obs_indices" is meant
        to be private. If no set of active indices has been specified, then
        all indices will be assumed active. If no input-target pairs have
        been specified at the time of execution of this method, 'None' will be
        returned.

        Returns
        -------
        active_obs_indices : array, shape ( N )
            The set of currently active indices
        """
        if self._active_obs_indices is not None:
            active_obs_indices = self._active_obs_indices
        elif self._targets is not None:
            active_obs_indices = np.arange(0, self.n_inputs)
        else:
            active_obs_indices = None

        return active_obs_indices

    def get_num_active_obs_indices(self):
        """Get the number of currently active observation indices.

        At any given time, only a subset of the input-target pairs may be
        "active" (i.e. used for defining the Gaussian Process). This function
        returns the number of currently active indices.

        Returns
        -------
        num_active_obs_indices : int
            The number of currently active indices
        """
        active_obs_indices = self.get_active_obs_indices()
        n_active = active_obs_indices.shape[0]

        return n_active

    def get_prob_mat(self):
        """Get the probability matrix indicating how likely each data instance
        to belong to the Gaussian process

        Returns
        -------
        prob_mat : array, shape ( n_instances, n_instances )
            Diagonal array. Each diagonal element indicates the probability
            that the corresponding target belongs to the Gaussian Process.
        """
        assert self._prob_mat is not None, "Attempt to get probability matrix,\
        but the matrix is None"
        
        return self._prob_mat

    def get_active_prob_mat(self):
        """Get the probability matrix indicating how likely each data instance
        to belong to the Gaussian process. Do not include the columns/rows
        corresponding to those data instances that are very unlikely to belong
        to the regressor.

        Returns
        -------
        active_prob_mat : array, shape ( n_instances, n_instances )
            Diagonal array. Each diagonal element indicates the probability
            that the corresponding target belongs to the Gaussian Process. Only
            entries that correspond to the active observations will be included            
        """
        assert self._prob_mat is not None, "Attempt to get probability matrix,\
        but the matrix is None"
        
        # Get the currently active observation indices
        active_obs_indices = self.get_active_obs_indices()
        n_active = active_obs_indices.shape[0]

        assert n_active > 0, "Number of active instances is not greater than 0"

        active_prob_mat = \
            self.sub_sample_mat(self._prob_mat, active_obs_indices)

        return active_prob_mat

    def compute_gram_mat(self):        
        assert self._inputs is not None, "No inputs"
        assert self._kernel is not None, "Empty kernel"

        self._gram_mat = compute_gram_mat(self._inputs, self._kernel)

    def sub_sample_mat(self, mat, indices):
        """Sub-sample a matrix using rows/columns specified in 'indices'

        Parameters
        ----------
        mat : array, shape ( N, N )
            Input array to sub-sample. Assumed to be symmetric

        indices : array, shape ( M )
            Collection of indices to sub-sample 'mat' along. E.g. if 'indices'
            is [3, 5], then the output matrix will be 2x2 and will consist of
            entries taken rows/cols 3 and 5.

        Returns
        -------
        sub_sampled_mat : array, shape ( M, M )
            Sub-sampled array. Symmetric
        """
        if mat.shape[0] != mat.shape[1]:
            raise ValueError("Matrix must be square")
        if np.min(indices) < 0:
            raise ValueError("Indices can not be negative")
        if np.max(indices) > mat.shape[0]-1:
            raise ValueError("Indices can not be larger than matrix extent")

        # Pre-allocate space for sub-sampled matrix
        sub_sampled_mat = np.zeros((indices.shape[0], indices.shape[0]))

        for i in np.arange(0, indices.shape[0]):
            for j in np.arange(i, indices.shape[0]):
                r = indices[i]
                c = indices[j]
                sub_sampled_mat[i, j] = sub_sampled_mat[j, i] = mat[r, c]

        return sub_sampled_mat

    def extract_vec(self, mat, col, rows):
        """Extract a column from a matrix.

        The returned vector has entries from 'mat' at the specified column
        and only has entries along the column at the specified rows

        Parameters
        ----------
        mat : array, shape ( N, N )
            The input matrix from which to extract the column

        col : int
            The column of the matrix to extract the vector from

        rows : array, shape ( M )
            The rows of 'mat' to populate the output vector with

        Returns
        -------
        vec : array, shape ( M )
            The extracted vector
        """
        if col < 0:
            raise ValueError("Specified column can not be negative")
        if col > mat.shape[1]:
            raise ValueError("Specified column can not be larger than matrix \
                             extent")
        if np.min(rows) < 0:
            raise ValueError("Indices can not be negative")
        if np.max(rows) > mat.shape[0]:
            raise ValueError("Indices can not be larger than matrix extent")

        # Allocate space for the output vector
        vec = np.zeros(rows.shape[0])

        for i in np.arange(0, rows.shape[0]):
            vec[i] = mat[rows[i], col]

        return vec

    def get_active_cov_mat(self):
        """Get the Gaussian process covariance matrix with respect to the
        active input-target pairs.

        Returns
        -------
        cov_mat : array, shape( n_active, n_active )
            Covariance matrix for Guassian Process regression. See Bishop's
            'Pattern Recognition and Machine Learning', equations 6.66 and
            6.67 for the role of the covariance matrix in Gaussin Process
            prediction. n_active indicates the number of active data instances.
            Note that in general the dimensions of the covariance matrix can be
            smaller than the number of data instances given that only some of
            the data instances may be set to active.
        """        
        assert self._inputs is not None, "Attempt to get active covariance\
        matrix, but inputs are None"
        assert self._precision is not None, "Attempt to get active covariance\
        matrix, but precision is None"
        assert self._kernel is not None, "Attempt to get active covariance\
        matrix, but kernel is None"
        assert self._prob_mat is not None, "Attempt to get active covariance\
        matrix, but probability matrix is None"

        # The probability matrix must be square and have the same number of
        # rows as data instances. 
        assert self._prob_mat.shape[0] == self._inputs.shape[0], "Attempt to\
        covariance matrix, but dimensions of probability matrix don't coincide\
        with the number of inputs"
            
        # Get the currently active observation indices
        active_obs_indices = self.get_active_obs_indices()
        n_active = active_obs_indices.shape[0]

        assert n_active > 0, "Number of active instances is not greater than 0"

        # If the Gram matrix is 'None', compute it now
        if self._gram_mat is None:
            self.compute_gram_mat()

        # Sub-sample the Gram matrix and probability matrix according to the
        # currently active set of indices
        gram_mat = self.sub_sample_mat(self._gram_mat, active_obs_indices)
        prob_mat = self.sub_sample_mat(self._prob_mat, active_obs_indices)

        assert self._precision > 0.0, "Precision is <= 0.0"
        if gram_mat.shape[0] != gram_mat.shape[1]:
            raise ValueError("Gram matrix is not square")
        if prob_mat.shape[0] != prob_mat.shape[1]:
            raise ValueError("Probability matrix is not square")

        self._active_cov_mat = gram_mat + \
            np.linalg.inv(self._precision*prob_mat)

        return self._active_cov_mat

    def get_active_inv_cov_mat(self):
        """Get the Gaussian process inverse covariance matrix with respect to
        the active input-target pairs.

        Returns
        -------
        inv_cov_mat : array, shape( n_active, n_active )
            Inverse covariance matrix for Guassian Process regression. n_active
            indicates the number of active data instances. Note that in general
            the dimensions of the covariance matrix can be smaller than the
            number of data instances given that only some of the data instances
            may be set to active.
        """
        assert self._active_cov_mat is not None, "Attempt to get inverse\
        covariance matrix, but covariance matrix is None"
        
        # Use SVD to compute the inverse of the Gaussian Process covariance matrix.
        # This is done to protect against numerical instability
        [U, S, Vh] = np.linalg.svd(self._active_cov_mat)

        inv_S = np.zeros((self._active_cov_mat.shape[0], \
                          self._active_cov_mat.shape[0]))
        for i in np.arange(0, self._active_cov_mat.shape[0]):
            if S[i] > np.spacing(1):
                inv_S[i, i] = 1.0/S[i]

        self._inv_cov_mat = np.dot(Vh.T, np.dot(inv_S, U.T))

        return self._inv_cov_mat

    def get_inv_gram_mat(self):
        """Get the inverse Gram matrix.

        Returns
        -------
        inv_gram_mat : array, shape( n_instances, n_instances )
            Inverse Gram matrix. n_instances is the number of input-target
            pairs.
        """
        assert self._gram_mat is not None, "Attempt to get inverse Gram\
        matrix, but Gram matrix is None"
        
        # Use SVD to compute the inverse of the Gram matrix. This is done to
        # protect against numerical instability
        [U, S, Vh] = np.linalg.svd(self._gram_mat)

        inv_S = np.zeros((self._gram_mat.shape[0], self._gram_mat.shape[0]))
        for i in np.arange(0, self._gram_mat.shape[0]):
            if S[i] > np.spacing(1):
                inv_S[i, i] = 1.0/S[i]

        self._inv_gram_mat = np.dot(Vh.T, np.dot(inv_S, U.T))

        return self._inv_gram_mat    

    def get_cov_mat(self):
        """Get the covariance matrix for the Gaussian procces.

        Returns
        -------
        cov_mat : array, shape( n_instances, n_instances )
            Covariance matrix for Guassian rocess regression. See Bishop's
            'Pattern Recognition and Machine Learning', equations 6.66 and
            6.67 for the role of the covariance matrix in Gaussin Process
            prediction. 
        """        
        assert self._inputs is not None, "Attempt to get covariance matrix but\
        inputs are None"
        assert self._precision is not None, "Attemp to get covariance matrix,\
        but precision is None"
        assert self._kernel is not None, "Attempt to get covariance matrix,\
        but kernel is None"
        assert self._prob_mat is not None, "Attemp to get covariance matrix,\
        but probability matrix is None"

        # The probability matrix must be square and have the same number of
        # rows as data instances. 
        assert self._prob_mat.shape[0] == self._inputs.shape[0], "Attempt to\
        covariance matrix, but dimensions of probability matrix don't coincide\
        with the number of inputs"
            
        self._cov_mat = self.get_gram_mat() + \
            self._precision*self.get_prob_mat()

        return self._cov_mat
