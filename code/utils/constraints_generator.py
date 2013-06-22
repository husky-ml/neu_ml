import pdb

import sys
import numpy as np
import networkx as nx

class ConstraintsGenerator:
    """Class handling creation and management of must-link and cannot-link
    constraints.

    Parameters
    ----------
    index_groups : list of arrays with shape ( n_group_instances ), optional
        Each element of 'index_groups' is an array containing indices
        representing instances within a particular group. It's assumed that the
        complete dataset is arranged in a matrix -- call it X. The row number
        of a given entry in X is an element of one of the 'index_groups'. A
        'group' represents, e.g., a data cluster, a segmented region in an
        image, etc.

    cardinality_thresh : integer, optional
        The must-link and cannot-link constraints are contained in a graph.
        This threshold specifies the maximum cardinality that any connected
        subgraph is allowed to have.
        
    must_link_dist_thresh : float, optional
        Distance threshold for must-link constraints. This is an optional
        input, and is only meaningful if 'tuple_groups' is also specified.
        Specification of these two will limit the must-link constraints to
        those pairs that are within 'must_link_dist_thresh' (Euclidean
        distance) of one another.

    cannot_link_dist_thresh : float, optional
        Distance threshold for cannot-link constraints. This is an optional
        input, and is only meaningful if 'tuple_groups' is also specified.
        Specification of these two will limit the cannot-link constraints to
        those pairs that are within 'cannot_link_dist_thresh' (Euclidean
        distance) of one another.

    tuple_groups : list of arrays with shape (n_group_instances, dim), optional
        Tuple groups correspond to 'index_groups'. I.e. a given 'index_group'
        entry reprents a particular row number in the complete data matrix, X,
        and the corresponding 'tuple_group' represents the actual row entry.
        This is an optional entry, and is only meaningful if 'dist_thresh' is
        also specified. Specification of these two will limit the constraints
        to those pairs that are within a specified (Euclidean) distance of one
        another.

    constraints : networkx graph, optional
        The constraints are encoded in a networkx graph, in which each graph
        node is a tuple from one of the 'groups', and each edge represents a
        constraint. The edges have a string attribute called 'constraint',
        which is either set to 'must_link' or 'cannot_link'.    
    """
    def __init__(self, index_groups=None, cardinality_thresh=sys.maxint,
                 must_link_dist_thresh=sys.float_info.max,
                 cannot_link_dist_thresh=sys.float_info.max, tuple_groups=None,
                 constraints=None):
        self._index_groups = index_groups        
        self._tuple_groups = tuple_groups
        if constraints is None:
            self._constraints = nx.Graph()
        else:
            self._constraints = constraints
        self._cardinality_thresh = cardinality_thresh
        self._must_link_dist_thresh = must_link_dist_thresh
        self._cannot_link_dist_thresh = cannot_link_dist_thresh

        # '_info_graph' is a fully connected graph that stores the inter-tuple
        # distances as well as the constraint type between instance IDs. To do
        # this, it's assumed that must-links exist between instances in the
        # same index group and cannot-links exist between instances belonging
        # to different index groups
        self._info_graph = None

    def add_nearest_possible_constraint(self, index):
        """Add a must-link or cannot-link constraint between the instance with
        the specified index and the closest nearby instance.

        If 'tuple_groups' is None, this function will return False. If the
        requested index is already a member of a subgraph that has a
        cardinality at the cardinality threshold, this function will return
        False. If a constraint is successfully added, this function will
        return True.

        Parameters
        ----------
        index, int
            Data index to create a constraint with.

        Returns
        -------
        constraint_added, bool
            True if a constraint was added and false otherwise
        """
        if self._tuple_groups is None:
            return False

        # Compute all the inter-tuple distances if necessary
        if self._info_graph is None:
            self._compute_distances()

        tested_nodes = []
        while len(tested_nodes) < self._info_graph.degree(index):
            # First find the closest adjacent node (tuple)
            min_dist = sys.float_info.max
            nearby_node = index
            for e in self._info_graph.edges_iter(index):
                n = e[1]
                # Test to see if the node has already been considered
                already_considered = False
                for t in xrange(0, len(tested_nodes)):
                    if n == tested_nodes[t]:
                        already_considered = True
                        break

                # Proceed if the node in question has not already been
                # considered
                if not already_considered:
                    if self._info_graph.edge[index][n]['dist'] < min_dist:
                        nearby_node = n
                        min_dist = self._info_graph.edge[index][n]['dist']
                        constraint = \
                            self._info_graph.edge[index][n]['constraint']

            # Now attempt to add the node
            if nearby_node != index:
                tested_nodes.append(nearby_node)
                if self.add_constraint(index, nearby_node, constraint):
                    return True

        return False
                
    def add_constraint(self, index1, index2, constraint_type):                            
        """Add a constraint to a graph provided no connected subgraph cardinality
        exceeds the cardinality threshold.

        Parameters
        ----------
        index1 : int
            One of the data matrix row indices between which to form a link

        index2 : int
            One of the data matrix row indices between which to form a link

        constraint_type : string
            Either "must_link" or "cannot_link". If the link is established, the
            graph edge will be assigned this designation
        
        Returns
        -------
        is_link_created : bool
            If the link is created, the return value will be 'True'. Otherwise, it
            will be 'False'
        """
        index1_exists = False
        index2_exists = False

        if index1 in self._constraints:
            index1_exists = True
        if index2 in self._constraints:
            index2_exists = True
        
        is_link_created = True
        # If the edge already exists do nothing an return 'False'
        if (index1, index2) in self._constraints.edges() or \
            (index2, index1) in self._constraints.edges():
            is_link_created = False

        # Proceed if the link doesn't already exist
        if is_link_created:
            # Add the edge (link)
            self._constraints.add_edge(index1, index2, constraint=constraint_type)

            # Now check that the added edge doesn't create a subgraph with a
            # cardinality that exceeds the max allowable subgraph cardinality
            connected_components = nx.connected_components(self._constraints)
            num_components = len(connected_components)
            for i in xrange(0, num_components):
                if len(connected_components[i]) > self._cardinality_thresh:
                    is_link_created = False
                    break

            # If the link created a subgraph that is too large, remove the link
            if not is_link_created:
                self._constraints.remove_edge(index1, index2)
                if not index1_exists:
                    self._constraints.remove_node(index1)
                if not index2_exists:
                    self._constraints.remove_node(index2)

        return is_link_created

    def _compute_distances(self):
        """Computes and stores all inter-tuple distances in a graph structure.
        """
        assert self._tuple_groups is not None, "Tuple groups not specified"
        assert self._index_groups is not None, "Index groups not specified"
        assert len(self._index_groups) == len(self._tuple_groups), "Number of\
        index groups must be same as number of tuple groups"

        if self._info_graph is None:
            self._info_graph = nx.Graph()

        num_groups = len(self._tuple_groups)
        for i in xrange(0, num_groups):
            num_tuples1 = len(self._tuple_groups[i])
            for j in xrange(0, num_groups):
                num_tuples2 = len(self._tuple_groups[j])
                for m in xrange(0, num_tuples1):
                    for n in xrange(0, num_tuples2):
                        if i != j or m != n:
                            t1 = self._tuple_groups[i][m]
                            t2 = self._tuple_groups[j][n]
                            dist = self._get_dist(t1, t2)
                            i1 = self._index_groups[i][m]
                            i2 = self._index_groups[j][n]                            
                            if i == j:
                                c = 'must_link'
                            else:
                                c = 'cannot_link'

                            self._info_graph.add_edge(i1, i2, dist=dist,
                                                     constraint=c)
                        
    def _get_dist(self, tuple1, tuple2):
        """Get the Euclidean distance between the two tuples.

        Parameters
        ----------
        tuple1 : array, shape( dim )
            One of the two tuples between which to compute the distance

        tuple2 : array, shape( dim )
            One of the two tuples between which to compute the distance

        Returns
        -------
        dist : float
            The Euclidean distance between the two tuples
        """
        diff = tuple1 - tuple2
        dist = np.sqrt(np.dot(diff, diff))

        return dist                
        

    def add_nearest_possible_must_link(self, index):
        """TODO: implement"""
        pass

    def add_nearest_possible_cannot_link(self, index):
        """TODO: implement"""
        pass

    def get_constraints(self):
        """TODO: Comment"""
        return self._constraints

    def get_num_constraints(self):
        """Get the total number of constraints including both must-link and
        cannot-link constraints.

        Returns
        -------
        num_constraints, int
            Total number of constraints
        """
        return len(self._constraints.edges())

    def get_num_must_links(self):
        """Get the total number of must-link constraints.

        Returns
        -------
        num_must_links, int
            Total number of must-link constraints
        """
        num = 0
        for i in xrange(0, self.get_num_constraints()):
            n1 = self._constraints.edges()[i][0]
            n2 = self._constraints.edges()[i][1]                
            if self._constraints.edge[n1][n2]['constraint'] == 'must_link':
                num += 1

        return num

    def get_num_cannot_links(self):
        """Get the total number of cannot-link constraints.

        Returns
        -------
        num_cannot_links, int
            Total number of must-link constraints
        """
        num = 0
        for i in xrange(0, self.get_num_constraints()):
            n1 = self._constraints.edges()[i][0]
            n2 = self._constraints.edges()[i][1]                
            if self._constraints.edge[n1][n2]['constraint'] == 'cannot_link':
                num += 1

        return num    

    def set_cardinality_thresh(self, cardinality_thresh):
        """Set the class instance's cardinality threshold

        Parameters
        ----------
        cardinality_thresh : int
            The must-link and cannot-link constraints are contained in a graph.
            This threshold specifies the maximum cardinality that any connected
        """
        self._cardinality_thresh = cardinality_thresh

    def set_constraints(self, constraints):
        """Set the class constraints.

        Parameters
        ----------
        constraints : networkx graph
            The constraints are encoded in a networkx graph, in which each
            graph node represents a data instance and each graph edge
            represencts a constraint. The edges have a string attribute called
            'constraint', which is either set to 'must_link' or 'cannot_link'.
        """
        # TODO: copy instead of assign?
        self._constraints = constraints

    def set_index_groups(self, index_groups):
        """Set the instance's index groups.

        Parameters
        ----------
        index_groups : list of arrays shaped ( n_group_instances ), optional
            Each element of 'index_groups' is an array containing indices
            representing instances within a particular group. It's assumed
            that the complete dataset is arranged in a matrix -- call it X.
            The row number of a given entry in X is an element of one of the
            'index_groups'. A 'group' represents, e.g., a data cluster, a
            segmented region in an image, etc.        
        """
        self._index_groups = index_groups

    def set_tuple_groups(self, tuple_groups):
        """Set the instance's tuple groups.

        Parameters
        ----------
        tuple_groups : list of arrays shaped (n_group_instances, dim), optional
            Tuple groups correspond to 'index_groups'. I.e. a given
            'index_group' entry reprents a particular row number in the
            complete data matrix, X, and the corresponding 'tuple_group'
            represents the actual row entry. This is an optional entry, and is
            only meaningful if 'dist_thresh' is also specified. Specification
            of these two will limit the constraints to those pairs that are
            within a specified (Euclidean) distance of one another.
        """
        self._tuple_groups = tuple_groups

def constraints_generator(index_groups, num_must_links=0, num_cannot_links=0,
                         cardinality_thresh=sys.maxint,
                         tuple_groups=None,
                         must_link_max_dist=sys.float_info.max,
                         cannot_link_max_dist=sys.float_info.max,
                         must_link_min_dist=0.0, cannot_link_min_dist=0.0):
    """Randomly generate must-link and/or cannot-link constraints among the
    elements specified in 'groups'.

    The function will attempt to produce as many must-link and cannot-link
    constraints as requested. However, it may not be possible to do so and
    still honor the specified cardinality and distance threshold values. In
    that case, the function will generate as many constraints as possible.

    Parameters
    ----------
    index_groups : list of arrays with shape ( n_group_instances )
        Each element of 'index_groups' is an array containing indices
        representing instances within a particular group. It's assumed that the
        complete dataset is arranged in a matrix -- call it X. The row number
        of a given entry in X is an element of one of the 'index_groups'. A
        'group' represents, e.g., a data cluster, a segmented region in an
        image, etc.

    num_must_links : integer
        Number of must-link constraints to generate. The must-link constraints
        will be spread uniformly across the groups.

    num_cannot_links : integer
        Number of cannot-link constraints to generate. The constraints will be
        spread uniformly across the groups.

    cardinality_thresh : integer, optional
        The must-link and cannot-link constraints are contained in a graph.
        This threshold specifies the maximum cardinality that any connected
        subgraph is allowed to have.

    tuple_groups : list of arrays with shape (n_group_instances, dim), optional
        Tuple groups correspond to 'index_groups'. I.e. a given 'index_group'
        entry reprents a particular row number in the complete data matrix, X,
        and the corresponding 'tuple_group' represents the actual row entry.
        This is an optional entry, and is only meaningful if 'dist_thresh' is
        also specified. Specification of these two will limit the constraints
        to those pairs that are within a specified (Euclidean) distance of one
        another.

    must_link_max_dist : float, optional
        Max distance threshold for must-link constraints. This is an optional
        input, and is only meaningful if 'tuple_groups' is also specified.
        Specification of these two will limit the must-link constraints to
        those pairs that are within 'must_link_max_dist' (Euclidean
        distance) of one another.

    cannot_link_max_dist : float, optional
        Max distance threshold for cannot-link constraints. This is an optional
        input, and is only meaningful if 'tuple_groups' is also specified.
        Specification of these two will limit the cannot-link constraints to
        those pairs that are within 'cannot_link_max_dist' (Euclidean
        distance) of one another.

    must_link_min_dist : float, optional
        Min distance threshold for must-link constraints. This is an optional
        input, and is only meaningful if 'tuple_groups' is also specified.
        Specification of these two will limit the must-link constraints to
        those pairs that are at least 'must_link_min_dist' (Euclidean
        distance) apart.

    cannot_link_min_dist : float, optional
        Min distance threshold for cannot-link constraints. This is an optional
        input, and is only meaningful if 'tuple_groups' is also specified.
        Specification of these two will limit the cannot-link constraints to
        those pairs that are at least 'cannot_link_min_dist' (Euclidean
        distance) of one another.

    Returns
    -------
    graph : networkx graph
        The constraints are encoded in a networkx graph, in which each graph
        node is a tuple from one of the 'groups', and each edge represents a
        constraint. The edges have a string attribute called 'constraint',
        which is either set to 'must_link' or 'cannot_link'.

    """
    if cardinality_thresh < 1:
        raise ValueError("Cardinality threshold must be greater than 1")

    if tuple_groups is not None:
        if len(index_groups) != len(tuple_groups):
            raise ValueError("Number of index groups not equal to number of\
                              tuple groups")
        else:
            for i in np.arange(0, len(index_groups)):
                if index_groups[i].shape[0] != tuple_groups[i].shape[0]:
                    raise ValueError("An index group has different number of\
                                     elements than the corresponding tuple\
                                     group")

    # Begin by getting all possible must links and all possible cannot-links.
    # We select from these lists
    possible_must_links = get_possible_must_links(index_groups, tuple_groups,
                                                  must_link_min_dist,
                                                  must_link_max_dist)

    possible_cannot_links = get_possible_cannot_links(index_groups,
                                                      tuple_groups,
                                                      cannot_link_min_dist,
                                                      cannot_link_max_dist)
    
    # Create a graph to hold the constraints. A graph is a natural structure
    # to both hold the constraints and to keep track of the cardinality of
    # subgraphs
    graph = nx.Graph()

    # Now generate the must-link constraints
    links_counter = 0
    inc = 0
    while links_counter < num_must_links and inc < len(possible_must_links):
        index1 = possible_must_links[inc][0]
        index2 = possible_must_links[inc][1]
        inc += 1

        if add_constraint_to_graph(graph, index1, index2, "must_link",
                                   cardinality_thresh):
            links_counter += 1

    # Generate the cannot-link constraints
    links_counter = 0
    inc = 0
    while links_counter < num_cannot_links and inc < \
        len(possible_cannot_links):
        index1 = possible_cannot_links[inc][0]
        index2 = possible_cannot_links[inc][1]
        inc += 1

        if add_constraint_to_graph(graph, index1, index2, "cannot_link",
                                   cardinality_thresh):
            links_counter += 1
        
    return graph

def add_constraint_to_graph(graph, index1, index2, constraint_type,
                            cardinality_thresh):
    """Add a constraint to a graph provided no connected subgraph cardinality
    exceeds the cardinality threshold.

    Parameters
    ----------
    graph : networkx Graph object
        The graph that keeps track of all the created links. If the requested
        link does not create a connected subgraph whose cardinality exceeds
        'cardinality_thresh', the link will be established. Otherwise, it will
        not.

    index1 : int
        One of the data matrix row indices between which to form a link

    index2 : int
        One of the data matrix row indices between which to form a link

    constraint_type : string
        Either "must_link" or "cannot_link". If the link is established, the
        graph edge will be assigned this designation
        
    cardinality_thresh : integer
        Indicates the maximum cardinality of any connected subgraph in the
        input graph. If the requested link would create a subgraph with a
        cardinality larger than this, the link will not be established.

    Returns
    -------
    is_link_created : bool
        If the link is created, the return value will be 'True'. Otherwise, it
        will be 'False'
    """
    index1_exists = False
    index2_exists = False

    if index1 in graph:
        index1_exists = True
    if index2 in graph:
        index2_exists = True

    is_link_created = True

    # If the edge already exists do nothing an return 'False'
    if (index1, index2) in graph.edges() or (index2, index1) in graph.edges():
        is_link_created = False

    # Proceed if the link doesn't already exist
    if is_link_created:
        # Add the edge (link)
        graph.add_edge(index1, index2, constraint=constraint_type)

        # Now check that the added edge doesn't create a subgraph with a
        # cardinality that exceeds the max allowable subgraph cardinality
        connected_components = nx.connected_components(graph)
        num_components = len(connected_components)
        for i in np.arange(0, num_components):
            if len(connected_components[i]) > cardinality_thresh:
                is_link_created = False
                break

        # If the link created a subgraph that is too large, remove the link
        if not is_link_created:
            graph.remove_edge(index1, index2)
            if not index1_exists:
                graph.remove_node(index1)
            if not index2_exists:
                graph.remove_node(index2)

    return is_link_created

def get_possible_must_links(index_groups, tuple_groups=None,
                            min_dist_thresh=0.0,
                            max_dist_thresh=sys.float_info.max):
    """Returns a randomly shuffled list of data index pairs representing all
    possible must-link constraints.

    Parameters
    ----------
    index_groups : list of arrays with shape ( n_group_instances )
        Each element of 'index_groups' is an array containing indices
        representing instances within a particular group. It's assumed that the
        complete dataset is arranged in a matrix -- call it X. The row number
        of a given entry in X is an element of one of the 'index_groups'. A
        'group' represents, e.g., a data cluster, a segmented region in an
        image, etc.

    tuple_groups : list of arrays with shape (n_group_instances, dim), optional
        Tuple groups correspond to 'index_groups'. I.e. a given 'index_group'
        entry reprents a particular row number in the complete data matrix, X,
        and the corresponding 'tuple_group' represents the actual row entry.
        This is an optional entry, and is only meaningful if 'dist_thresh' is
        also specified. Specification of these two will limit the constraints
        to those pairs that are within a specified (Euclidean) distance of one
        another.

    min_dist_thresh : float, optional
        Min distance threshold. This is an optional input, and is only
        meaningful if 'tuple_groups' is also specified. Specification of these
        two will limit the constraints to those pairs that at least
        'min_dist_thresh' (Euclidean distance) apart.

    max_dist_thresh : float, optional
        Max distance threshold. This is an optional input, and is only
        meaningful if 'tuple_groups' is also specified. Specification of these
        two will limit the constraints to those pairs that at most
        'min_dist_thresh' (Euclidean distance) apart.

    Returns
    -------
    possible_must_links : list of pairs of integeters
        Contains all possible index pairs representing must-link constraints
    """
    num_groups = len(index_groups)

    possible_must_links = []
    for i in np.arange(0, num_groups):
        for j in np.arange(0, len(index_groups[i])):
            for k in np.arange(j+1, len(index_groups[i])):
                if tuple_groups is None:
                    possible_must_links.append((index_groups[i][j],
                                                index_groups[i][k]))
                elif tuple_groups is not None:
                    dist = get_dist(tuple_groups[i][j], tuple_groups[i][k])
                    if dist >= min_dist_thresh and dist <= max_dist_thresh:
                        possible_must_links.append((index_groups[i][j],
                                                    index_groups[i][k]))

    np.random.shuffle(possible_must_links)

    return possible_must_links

def get_possible_cannot_links(index_groups, tuple_groups=None,
                              min_dist_thresh=0.0,
                              max_dist_thresh=sys.float_info.max):
    """Returns a randomly shuffled list of data index pairs representing all
    possible cannot-link constraints.

    Parameters
    ----------
    index_groups : list of arrays with shape ( n_group_instances )
        Each element of 'index_groups' is an array containing indices
        representing instances within a particular group. It's assumed that the
        complete dataset is arranged in a matrix -- call it X. The row number
        of a given entry in X is an element of one of the 'index_groups'. A
        'group' represents, e.g., a data cluster, a segmented region in an
        image, etc.

    tuple_groups : list of arrays with shape (n_group_instances, dim), optional
        Tuple groups correspond to 'index_groups'. I.e. a given 'index_group'
        entry reprents a particular row number in the complete data matrix, X,
        and the corresponding 'tuple_group' represents the actual row entry.
        This is an optional entry, and is only meaningful if 'dist_thresh' is
        also specified. Specification of these two will limit the constraints
        to those pairs that are within a specified (Euclidean) distance of one
        another.

    min_dist_thresh : float, optional
        Min istance threshold. This is an optional input, and is only
        meaningful if 'tuple_groups' is also specified. Specification of these
        two will limit the constraints to those pairs that are at least
        'min_dist_thresh' (Euclidean distance) apart.

    max_dist_thresh : float, optional
        Max distance threshold. This is an optional input, and is only
        meaningful if 'tuple_groups' is also specified. Specification of these
        two will limit the constraints to those pairs that are at most
        'max_dist_thresh' (Euclidean distance) apart.

    Returns
    -------
    possible_cannot_links : list of pairs of integeters
        Contains all possible index pairs representing must-link constraints
    """
    num_groups = len(index_groups)

    possible_cannot_links = []
    for i in np.arange(0, num_groups):
        for j in np.arange(i+1, num_groups):
            for m in np.arange(0, len(index_groups[i])):
                for n in np.arange(0, len(index_groups[j])):
                    if tuple_groups is None:
                        possible_cannot_links.append((index_groups[i][m],
                                                      index_groups[j][n]))
                    elif tuple_groups is not None:
                        dist = get_dist(tuple_groups[i][m], tuple_groups[j][n])
                        if dist >= min_dist_thresh and dist <= max_dist_thresh:
                            possible_cannot_links.append((index_groups[i][m],
                                                          index_groups[j][n]))

    np.random.shuffle(possible_cannot_links)

    return possible_cannot_links

def get_dist(tuple1, tuple2):
    """Get the Euclidean distance between the two tuples.

    Parameters
    ----------
    tuple1 : array, shape( dim )
        One of the two tuples between which to compute the distance

    tuple2 : array, shape( dim )
        One of the two tuples between which to compute the distance

    Returns
    -------
    dist : float
        The Euclidean distance between the two tuples
    """
    diff = tuple1 - tuple2
    dist = np.sqrt(np.dot(diff, diff))

    return dist
