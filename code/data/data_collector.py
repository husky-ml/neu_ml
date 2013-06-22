import pdb

import numpy as np

class DataCollector:
    """Data collection class used to record data during algorithm execution for
    subsequent analysis.

    There are three concepts used by this class: 'trial', 'run', and
    'iteration'. A 'trial' exists at the level of a particular algorithm
    execution and corresponds to a given set of algorithm parameter settings,
    e.g. A 'run' is a particular execution of a 'trial'. It's assumed that
    generally you will get different data for a given trial, even for the same
    input data (consider random initializations, for example). Finally, an
    'iteration' corresponds to the data at a particular iteration of algorithm
    execution.

    As an example, K-means run with K=2 and K=3 represent two different trials.
    K-means run five times with K=2 represents five runs of the K=2 trial. The
    cluster assignments at iteration 3, for run two, and trial K=2 represent
    data at a specific iteration for a given run and trial.

    You may be interested in running an algorithm several times for different
    parameter settings. An instance of this class will facilitate data
    recording.

    Before you collect any data, you must first call 'add_new_trial' to begin a
    new data collection trial.
    """
    
    def __init__(self):
        self._cluster_assignments = []
        self._trial_descriptions = []

    def set_cluster_assignments(self, assignments, trial=0, run=0,
                                iteration=0):
        """Set cluster assigments for a given trial, run, and iteration.

        Parameters
        ----------
        assignments : array, shape ( n_instances )
            An array of integer values indicating cluster assignments.

        trial : integer, optional
            Indicates which trial 'assignments' corresponds to. The first trial
            is indexed at 0, the second at 1, etc.

        run : integer, optional
            Indicates which run 'assignments' corresponds to. The first run
            is indexed at 0, the second at 1, etc.

        iteration : integer, optional
            Indicates which iteration 'assignments' corresponds to. The first
            iteration is indexed at 0, the second at 1, etc.
        """
        if trial >= self.get_num_trials():
            raise ValueError("Trial does not exist")
        if run >= self.get_num_runs(trial):
            raise ValueError("Run does not exist")
        if iteration >= self.get_num_iterations(trial, run):
            raise ValueError("Iteration does not exist")

        self._cluster_assignments[trial][run][iteration] = assignments

    def add_new_trial(self):
        """Add a new trial to the data collector
        """
        self._cluster_assignments.append([])

        # Also append a blank description of this trial to the list of
        # descriptions
        self._trial_descriptions.append({})

    def add_new_run(self, trial=0):
        """Add a new run for the specified trial.

        Parameters
        ----------
        trial : integer, optional
            The trial for which to add the new run
        """

        if trial >= self.get_num_trials:
            raise ValueError("Requested trial does not exist")

        self._cluster_assignments[trial].append([])

    def add_new_iteration(self, trial=0, run=0):
        """Add a new iteration for the specified trial and run.

        Parameters
        ----------
        trial : integer, optional
            The trial for which to add a new iteration

        run : integer, optional
            The run for which to add a new iteration
        """
        if trial >= self.get_num_trials:
            raise ValueError("Requested trial does not exist")
        if run >= self.get_num_runs(trial):
            raise ValueError("Requested run does not exist")

        self._cluster_assignments[trial][run].append([])

    def get_cluster_assignments(self, trial=0, run=0, iteration=0):
        """Get the cluster assignments for the specified trial, run, and,
        iteration.

        Parameters
        ----------
        trial : integer, optional
            An index indicating the trial number

        run : integer, optional
            An index indicating the run number

        iteration : integer, optional
            An index indicating the iteration number

        Returns
        -------
        assignment : array, shape ( n_instances )
            An array of indices indicatint cluster assignments for the
            corresponding data instances.
        """

        return self._cluster_assignments[trial][run][iteration]

    def get_num_trials(self):
        """Get the number of stored trials.

        Returns
        -------
        num_trials : integer
            The number of stored trials. 
        """

        return len(self._cluster_assignments)

    def get_num_runs(self, trial=0):
        """Get the number of stored runs for the specified trial.

        Parameters
        ----------
        trial : integer
            An index indicating which trial to return the number of runs for

        Returns
        -------
        num_runs : integer
            The number of stored runs for the specified trial. 
        """
        if trial >= self.get_num_trials():
            raise ValueError("Requested trial does not exist")
        
        return len(self._cluster_assignments[trial])

    def get_num_iterations(self, trial=0, run=0):
        """Get the number of stored iterations for the specified trial and run

        Parameters
        ----------
        trial : integer
            An index indicating which trial to return the number of iterations
            for

        run : integer
            An index indicating which run to return the number of iterations
            for

        Returns
        -------
        num_iterations : integer
            The number of stored iterations for the specified trial and run
        """
        if trial >= self.get_num_trials():
            raise ValueError("Requested trial does not exist")

        if run >= self.get_num_runs(trial):
            raise ValueError("Requested run does not exist")
        
        return len(self._cluster_assignments[trial][run])

    def set_trial_description(self, trial, description):
        """Set a description of the a specified trial.

        This is useful for documenting the settings used to produce the data
        for a given trial.

        Parameters
        ----------        
        trial : integer
            An index indicating the trial number

        description : dictionary
            A dictionary-based descripiton of a given trial. The dictionary
            entries can change from trial to trial. The onus is on the user to
            keep track of dictionary entries.
        """
        num_trials = self.get_num_trials()

        if trial >= num_trials:
            raise ValueError("Trial does not exist")

        self._trial_descriptions[trial] = description

    def get_trial_description(self, trial):
        """Get the description for the specified trial.

        This is useful for documenting the settings used to produce the data
        for a given trial.

        Parameters
        ----------
        trial, integer
            An index indicating the trial number

        Returns
        -------
        description, dictionary
            A dictionary-based descripiton of a given trial. The dictionary
            entries can change from trial to trial. The onus is on the user to
            keep track of dictionary entries.
        """
        num_descriptions = len(self._trial_descriptions)

        if trial > num_descriptions:
            raise ValueError("Requested trial does not exist")

        return self._trial_descriptions[trial]

    def get_latest_cluster_assignment_trial(self):
        """Get the trial index corresponding the 'latest' cluster assignment
        trial (the one assumed to be added most recently).

        Returns
        -------
        trial, integer
            Index of the latest cluster assignment trial
        """
        trial = len(self._cluster_assignments)-1

        return trial

    def get_latest_cluster_assignment_run(self, trial):
        """Get the run index corresponding the 'latest' cluster assignment
        run (the one assumed to be added most recently) for the specified trial.

        Parameters
        ----------
        trial, integer
            Index indicating which trial to get the latest run for.

        Returns
        -------
        run, integer
            Index of the latest cluster assignment run for the specified trial
        """
        num_trials = len(self._cluster_assignments)

        if trial == num_trials:
            raise ValueError("Specified trial does not exist")

        run = len(self._cluster_assignments[trial])-1

        return run

    def get_latest_cluster_assignment_iteration(self, trial, run):
        """Get the run index corresponding the 'latest' cluster assignment
        iteration (the one assumed to be added most recently) for the
        specified trial and run.

        Parameters
        ----------
        trial, integer
            Index indicating which trial to get the latest iteration for.

        run, integer
            Index indicating which run to get the latest iteration for.

        Returns
        -------
        iteration, integer
            Index of the latest cluster assignment iteration for the specified
            trial and run
        """
        if trial >= self.get_num_trials():
            raise ValueError("Specified trial does not exist")
        if run >= self.get_num_runs(trial):
            raise ValueError("Specified run does not exist")
        
        iteration = self.get_num_iterations(trial, run) - 1

        return iteration
