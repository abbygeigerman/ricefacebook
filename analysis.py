import matplotlib.pyplot as plt
import pylab
import types
import time
import math
import copy
import itertools
import collections
import numpy
import random
import math
import bookgraphs
import autograder

## Graph functions

def read_graph(filename):
    """
    Read a graph from a file.  The file is assumed to hold a graph
    that was written via the write_graph function.

    Arguments:
    filename -- name of file that contains the graph

    Returns:
    The graph that was stored in the input file.
    """
    with open(filename) as f:
        g = eval(f.read())
    return g

def write_graph(g, filename):
    """
    Write a graph to a file.  The file will be in a format that can be
    read by the read_graph function.

    Arguments:
    g        -- a graph
    filename -- name of the file to store the graph

    Returns:
    None
    """
    with open(filename, 'w') as f:
        f.write(repr(g))

def copy_graph(g):
    """
    Return a copy of the input graph, g

    Arguments:
    g -- a graph

    Returns:
    A copy of the input graph that does not share any objects.
    """
    return copy.deepcopy(g)

## Timing functions

def time_func(f, args=[], kw_args={}):
    """
    Times one call to f with args, kw_args.

    Arguments:
    f       -- the function to be timed
    args    -- list of arguments to pass to f
    kw_args -- dictionary of keyword arguments to pass to f.

    Returns: 
    a tuple containing the result of the call and the time it
    took (in seconds).

    Example:

    >>> def sumrange(low, high):
            sum = 0
            for i in range(low, high):
                sum += i
            return sum
    >>> time_func(sumrange, [82, 35993])
    (647726707, 0.01079106330871582)
    >>> 
    """
    start_time = time.time()
    result = f(*args, **kw_args)
    end_time = time.time()

    return (result, end_time - start_time)

## Plotting functions

def show():
    """
    Do not use this function unless you have trouble with figures.

    It may be necessary to call this function after drawing/plotting
    all figures.  If so, it should only be called once at the end.

    Arguments:
    None

    Returns:
    None
    """
    plt.show()

def plot_dist_linear(data, title, xlabel, ylabel, filename=None):
    """
    Plot the distribution provided in data as a bar plot on a linear
    scale.

    Arguments: 
    data     -- dictionary which will be plotted with the keys
                on the x axis and the values on the y axis
    title    -- title label for the plot
    xlabel   -- x axis label for the plot
    ylabel   -- y axis label for the plot
    filename -- optional name of file to which plot will be
                saved (in png format)

    Returns:
    None
    """
    _plot_dist(data, title, xlabel, ylabel, False, filename)
    show()

def plot_dist_loglog(data, title, xlabel, ylabel, filename=None):
    """
    Plot the distribution provided in data as a scatter plot on a
    loglog scale.

    Arguments: 
    data     -- dictionary which will be plotted with the keys
                on the x axis and the values on the y axis
    title    -- title label for the plot
    xlabel   -- x axis label for the plot
    ylabel   -- y axis label for the plot
    filename -- optional name of file to which plot will be
                saved (in png format)

    Returns:
    None
    """
    _plot_dist(data, title, xlabel, ylabel, True, filename)

def _pow_10_round(n, up=True):
    """
    Round n to the nearest power of 10.

    Arguments:
    n  -- number to round
    up -- round up if True, down if False

    Returns:
    rounded number
    """
    if up:
        return 10 ** math.ceil(math.log(n, 10))
    else:
        return 10 ** math.floor(math.log(n, 10))
        
def _plot_dist(data, title, xlabel, ylabel, scatter, filename=None):
    """
    Plot the distribution provided in data.

    Arguments: 
    data     -- dictionary which will be plotted with the keys
                on the x axis and the values on the y axis
    title    -- title label for the plot
    xlabel   -- x axis label for the plot
    ylabel   -- y axis label for the plot
    scatter  -- True for loglog scatter plot, False for linear bar plot
    filename -- optional name of file to which plot will be
                saved (in png format)

    Returns:
    None
    """
    ### Check that the data is a dictionary
    if not isinstance(data, dict):
        msg = "data must be a dictionary, not {0}".format(type(data).__name__)
        raise TypeError(msg)

    ### Create a new figure
    fig = pylab.figure()

    ### Plot the data
    if scatter:
        _plot_dict_scatter(data)
    else:
        _plot_dict_bar(data, 0)
    
    ### Label the plot
    pylab.title(title)
    pylab.xlabel(xlabel)
    pylab.ylabel(ylabel)

    ### Draw grid
    gca = pylab.gca()
    gca.yaxis.grid(True)
    gca.xaxis.grid(False)

    if scatter:
        ### Use loglog scale
        gca.set_xscale('log')
        gca.set_yscale('log')
        gca.set_xlim([_pow_10_round(min([x for x in data.keys() if x > 0]), False), 
                      _pow_10_round(max(data.keys()))])
        gca.set_ylim([_pow_10_round(min([x for x in data.values() if x > 0]), False), 
                      _pow_10_round(max(data.values()))])

    ### Show the plot
    fig.show()

    ### Save to file
    if filename:
        pylab.savefig(filename)

def plot_lines(data, title, xlabel, ylabel, labels=None, filename=None):
    """
    Plot a line graph with the provided data.

    Arguments: 
    data     -- a list of dictionaries, each of which will be plotted 
                as a line with the keys on the x axis and the values on
                the y axis.
    title    -- title label for the plot
    xlabel   -- x axis label for the plot
    ylabel   -- y axis label for the plot
    labels   -- optional list of strings that will be used for a legend
                this list must correspond to the data list
    filename -- optional name of file to which plot will be
                saved (in png format)

    Returns:
    None
    """
    ### Check that the data is a list
    if not isinstance(data, list):
        msg = "data must be a list, not {0}".format(type(data).__name__)
        raise TypeError(msg)

    ### Create a new figure
    fig = pylab.figure()

    ### Plot the data
    if labels:
        mylabels = labels[:]
        for i in range(len(data)-len(labels)):
            mylabels.append("")
        for d, l in zip(data, mylabels):
            _plot_dict_line(d, l)
        # Add legend
        pylab.legend(loc='best')
        gca = pylab.gca()
        legend = gca.get_legend()
        pylab.setp(legend.get_texts(), fontsize='medium')
    else:
        for d in data:
            _plot_dict_line(d)

    ### Set the lower y limit to 0 or the lowest number in the values
    mins = [min(l.values()) for l in data]
    ymin = min(0, min(mins))
    pylab.ylim(ymin=ymin)

    ### Label the plot
    pylab.title(title)
    pylab.xlabel(xlabel)
    pylab.ylabel(ylabel)

    ### Draw grid lines
    pylab.grid(True)

    ### Show the plot
    fig.show()

    ### Save to file
    if filename:
        pylab.savefig(filename)

def _dict2lists(data):
    """
    Convert a dictionary into a list of keys and values, sorted by
    key.  

    Arguments:
    data -- dictionary

    Returns:
    A tuple of two lists: the first is the keys, the second is the values
    """
    xvals = list(data.keys())
    xvals.sort()
    yvals = []
    for x in xvals:
        yvals.append(data[x])
    return xvals, yvals

def _plot_dict_line(d, label=None):
    """
    Plot data in the dictionary d on the current plot as a line.

    Arguments:
    d     -- dictionary
    label -- optional legend label

    Returns:
    None
    """
    xvals, yvals = _dict2lists(d)
    if label:
        pylab.plot(xvals, yvals, label=label)
    else:
        pylab.plot(xvals, yvals)

def _plot_dict_bar(d, xmin=None, label=None):
    """
    Plot data in the dictionary d on the current plot as bars. 

    Arguments:
    d     -- dictionary
    xmin  -- optional minimum value for x axis
    label -- optional legend label

    Returns:
    None
    """
    xvals, yvals = _dict2lists(d)
    if xmin == None:
        xmin = min(xvals) - 1
    else:
        xmin = min(xmin, min(xvals) - 1)
    if label:
        pylab.bar(xvals, yvals, align='center', label=label)
        pylab.xlim([xmin, max(xvals)+1])
    else:
        pylab.bar(xvals, yvals, align='center')
        pylab.xlim([xmin, max(xvals)+1])

def _plot_dict_scatter(d):
    """
    Plot data in the dictionary d on the current plot as points. 

    Arguments:
    d     -- dictionary

    Returns:
    None
    """
    xvals, yvals = _dict2lists(d)
    pylab.scatter(xvals, yvals)
    
## Provided Functions
def remove_edges(g, edgelist):
    """
    Remove the edges in edgelist from the graph g.

    Arguments:
    g -- undirected graph
    edgelist - list of edges in g to remove

    Returns:
    None
    """
    for edge in edgelist:
        (u, v) = tuple(edge)
        g[u].remove(v)
        g[v].remove(u)        

def bfs(g, startnode):
    """
    Perform a breadth-first search on g starting at node startnode.

    Arguments:
    g -- undirected graph
    startnode - node in g to start the search from

    Returns:
    d -- distances from startnode to each node.
    n -- number of shortest paths from startnode to each node.
    """

    # Initiating dictionaries.
    d = {}
    n = {}
    q = collections.deque()

    # Set all distances to infinity.
    for i in g.keys():
        d[i] = float("inf")

    # Setting up the initial node's properties.
    d[startnode] = 0
    n[startnode] = 1

    q.append(startnode)

    while len(q) > 0:
        j = q.popleft()

        # For every neighbor of j.
        for h in g[j]:
            if d[h] == float("inf"):
                d[h] = d[j] + 1
                n[h] = n[j]
                q.append(h)
            elif d[h] == d[j] + 1:
                n[h] = n[h] + n[j]

    return d, n

def connected_components(g):
    """
    Find all connected components in g.

    Arguments:
    g -- undirected graph

    Returns:
    A list of sets where each set is all the nodes in
    a connected component.
    """
    # Initially we have no components and all nodes remain to be
    # explored.
    components = []
    remaining = set(g.keys())

    while remaining:
        # Randomly select a remaining node and find all nodes
        # connected to that node
        node = random.choice(list(remaining))
        distances = bfs(g, node)[0]
        visited = set()
        for i in remaining:
            if distances[i] != float('inf'):
                visited.add(i)
        components.append(visited)

        # Remove all nodes in this component from the remaining
        # nodes
        remaining -= visited

    return components

def gn_graph_partition(g):
    """
    Partition the graph g using the Girvan-Newman method.

    Requires connected_components, shortest_path_edge_betweenness, and
    compute_q to be defined.  This function assumes/requires these
    functions to return the values specified in the homework handout.

    Arguments:
    g -- undirected graph

    Returns:
    A list of tuples where each tuple contains a Q value and a list of
    connected components.
    """
    ### Start with initial graph
    c = connected_components(g)
    q = autograder.compute_q(g, c)
    partitions = [(q, c)]

    ### Copy graph so we can partition it without destroying original
    newg = copy_graph(g)

    ### Iterate until there are no remaining edges in the graph
    while True:
        ### Compute betweenness on the current graph
        btwn = autograder.shortest_path_edge_betweenness(newg)
        if not btwn:
            ### No information was computed, we're done
            break

        ### Find all the edges with maximum betweenness and remove them
        maxbtwn = max(btwn.values())
        maxedges = [edge for edge, b in btwn.items() if b == maxbtwn]
        remove_edges(newg, maxedges)

        ### Compute the new list of connected components
        c = connected_components(newg)
        if len(c) > len(partitions[-1][1]):
            ### This is a new partitioning, compute Q and add it to
            ### the list of partitions.
            q = autograder.compute_q(g, c)
            partitions.append((q, c))

    return partitions


### Use the following function to read the
### 'rice-facebook-undergrads.txt' file and turn it into an attribute
### dictionary.

def read_attributes(filename):
    """
    Code to read student attributes from the file named filename.
    
    The attribute file should consist of one line per student, where
    each line is composed of student, college, year, major.  These are
    all anonymized, so each field is a number.  The student number
    corresponds to the node identifier in the Rice Facebook graph.

    Arguments:
    filename -- name of file storing the attributes

    Returns:
    A dictionary with the student numbers as keys, and a dictionary of
    attributes as values.  Each attribute dictionary contains
    'college', 'year', and 'major' as keys with the obvious associated
    values.
    """
    attributes = {}
    with open(filename) as f:
        for line in f:
            # Split line into student, college, year, major
            fields = line.split()
            student = int(fields[0])
            college = int(fields[1])
            year    = int(fields[2])
            major   = int(fields[3])
            
             # Store student in the dictionary
            attributes[student] = {'college': college,
                                   'year': year,
                                   'major': major}
    return attributes

class LinAl(object):
    """
    Contains code for various linear algebra data structures and operations.
    """

    @staticmethod
    def zeroes(m, n):
        """
        Returns a matrix of zeroes with dimension m x n.
        ex: la.zeroes(3,2) -> [[0,0],[0,0],[0,0]]
        """

        return [[0] * n for i in range(m)]

    @staticmethod
    def trace(matrix):
        """
        Returns the trace of a square matrix. Assumes valid input matrix.
        ex: la.trace([[1,2],[-1,0]]) -> 1.0
        """

        if len(matrix[0]) == 0:
            return 0.0
        
        return float(sum(matrix[i][i] for i in range(len(matrix))))

    @staticmethod
    def transpose(matrix):
        """
        Returns the transpose of a matrix. Assumes valid input matrix.
        ex: la.transpose([[1,2,3],[4,5,6]]) -> [[1,4],[2,5],[3,6]]
        """

        res = [[0] * len(matrix) for i in range(len(matrix[0]))]

        for i in range(len(matrix[0])):
            for j in range(len(matrix)):
                res[i][j] = matrix[j][i]

        return res

    @staticmethod
    def dot(a, b):
        """
        Returns the dot product of two n x 1 vectors. Assumes valid input vectors.
        ex: la.dot([1,2,3], [3,-1,4]) -> 13.0
        """

        if len(a) != len(b):
            raise Exception("Input vectors must be of same length, not %d and %d" % (len(a), len(b)))

        return float(sum([a[i] * b[i] for i in range(len(a))]))

    @staticmethod
    def multiply(A, B):
        """
        Returns the matrix product of A and B. Assumes valid input matrices.
        ex: la.multiply([[1,2],[3,4]], [[-3,4],[2,-1]]) -> [[1.0,2.0],[-1.0,8.0]]
        """

        if len(A[0]) != len(B):
            raise Exception("Matrix dimensions do not match for matrix multiplication: %d x %d and %d x %d" % (len(A), len(A[0]), len(B), len(B[0])))

        result = [[0] * len(B[0]) for i in range(len(A))]

        for i in range(len(A)):
            for j in range(len(B[0])):

                result[i][j] = LinAl.dot(A[i], LinAl.transpose(B)[j])

        return result

    @staticmethod
    def sum(matrix):
        """
        Returns the sum of all the elements in matrix. Assumes valid input matrix.
        ex: la.sum([[1,2],[3,4]]) -> 10.0
        """

        return float(sum([sum(row) for row in matrix]))

    @staticmethod
    def multiply_by_val(matrix, val):
        """
        Returns the result of multiply matrix by a real number val. Assumes valid
        imput matrix and that val is a real number.
        """

        new_mat = LinAl.zeroes(len(matrix), len(matrix[0]))
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                new_mat[i][j] = val * matrix[i][j]
        return new_mat

def test_part_2():
    '''
    Inputs: none
    Outputs: none
    
    Prints the results of calling compute_flow and shortest_edge_betweenness on
    graphs from canvas and other hardcoded graphs. Expected results in Canvas and 
    in Comp182 OneNote Notebook. 
    '''

    ## Testing compute_flow
    graph3_14 = bookgraphs.fig3_14g
    #print(bfs(graph3_14, 1))
    
    canvas_graph_one = {0: {1, 2}, 1: {0,3}, 2:{0,3,4}, 3: {1,2,5}, 4: {2,5,6}, 5:{3,4}, 6:{4}}
    
    # print("Dist w/ start node 1 " , dist1)
    # print("Npaths w/ start node 1 ", npaths1)
    dist0, npaths0 = bfs(canvas_graph_one, 0)
    print("Node 0: ", dist0, npaths0)


    dist1, npaths1 = bfs(canvas_graph_one, 1)
    print("1 ", dist1, npaths1)
    #print("Compute flow 1", autograderhw4.compute_flow(canvas_graph_one, dist1, npaths1))

    dist2, npaths2 = bfs(canvas_graph_one, 2)
    print("2 ", dist2, npaths2)
    #print("Compute flow 2", autograderhw4.compute_flow(canvas_graph_one, dist2, npaths2))

    dist6, npaths6 = bfs(canvas_graph_one, 6)
    print("6 ", dist6, npaths6)
    #print("Compute flow 6", autograderhw4.compute_flow(canvas_graph_one, dist6, npaths6))

    print("SELF-CHECKED:")
    dist3, npaths3 = bfs(canvas_graph_one, 3)
    print("3 ", dist3, npaths3)
    #print("Compute flow 3", autograderhw4.compute_flow(canvas_graph_one, dist3, npaths3))

    dist4, npaths4 = bfs(canvas_graph_one, 4)
    print("4 ", dist4, npaths4)
    #print("Compute flow 4", autograderhw4.compute_flow(canvas_graph_one, dist4, npaths4))

    dist5, npaths5 = bfs(canvas_graph_one, 5)
    print("5 ", dist5, npaths5)
    #print("Compute flow 5", autograderhw4.compute_flow(canvas_graph_one, dist5, npaths5))

    graph3_15 = bookgraphs.fig3_15g
    dist, npaths = bfs(graph3_15, 'A')
    print(autograderhw4.compute_flow(graph3_18, dist, npaths))

    # Testing shortest_path_edge_betweenness
    print(autograderhw4.shortest_path_edge_betweenness(graph3_14))
    print(autograderhw4.shortest_path_edge_betweenness(graph3_15))
    graph_one = {"a": set(["b","c"]), "b": set(["a"]), "c": set(["a"])}

    compute_flow_graph_one_sols = {frozenset(["a","b"]): 1, frozenset(["a","c"]): 1}
    shortest_path_edge_betweenness_sols = {frozenset(["a","b"]): 4, frozenset(["a","c"]): 4}
    print(autograderhw4.shortest_path_edge_betweenness(graph_one))


    print("Betweenness ", autograderhw4.shortest_path_edge_betweenness(canvas_graph_one))

    g2 = {0:{1,2,3,4}, 1:{0,2,3,4}, 2:{0,1,3,4}, 3:{0,1,2,4}, 4:{0,1,2,3}}
    print("Shortest path edge betweenness complete graph: ", autograderhw4.shortest_path_edge_betweenness(g2))

    g3 = {0:{1,4}, 1:{0,2}, 2:{4,3,1}, 3:{2}, 4:{0,2}}
    distg3_0, pathsg3_0 = bfs(g3, 0)
    print("Compute flow w/ 0: ", autograderhw4.compute_flow(g3, distg3_0, pathsg3_0))
    distg3_1, pathsg3_1 = bfs(g3, 1)
    print("Compute flow w/ 1: ", autograderhw4.compute_flow(g3, distg3_1, pathsg3_1))
    distg3_2, pathsg3_2 = bfs(g3, 2)
    print("Compute flow w/ 2: ", autograderhw4.compute_flow(g3, distg3_2, pathsg3_2))
    distg3_3, pathsg3_3 = bfs(g3, 3)
    print("Compute flow w/ 3: ", autograderhw4.compute_flow(g3, distg3_3, pathsg3_3))
    distg3_4, pathsg3_4 = bfs(g3, 4)
    print("Compute flow w/ 4: ", autograderhw4.compute_flow(g3, distg3_4, pathsg3_4))

    print("Shortest path edge betweenness g3: ", autograderhw4.shortest_path_edge_betweenness(g3))

def test_part_3():
    '''
    Inputs: none
    Outputs: none
    
    Prints results of compute_q on graphs in canvas (expected results in canvas)
    '''
    graph3_14 = bookgraphs.fig3_14g
    graph3_15 = bookgraphs.fig3_15g

    list_one = [set([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])]
    list_two = [set([8, 9, 10, 11, 12, 13, 14]), set([1, 2, 3, 4, 5, 6, 7])]
    list_three = [set([9, 10, 11]), set([4, 5, 6]), set([8]), set([7]), set([1, 2, 3]), set([12, 13, 14])]
    list_four = [set([7]), set([14]), set([4]), set([12]), set([2]), set([6]), set([3]), set([9]), set([5]), set([13]), set([8]), set([1]), set([10]), set([11])]
    print("Example 2.1 ", autograderhw4.compute_q(graph3_14, list_one))
    print("Example 2.2 ", autograderhw4.compute_q(graph3_14, list_two))
    print("Example 2.3 ", autograderhw4.compute_q(graph3_14, list_three))
    print("Example 2.4 ", autograderhw4.compute_q(graph3_14, list_four))

    list_one_15 = [set([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])]
    list_two_15 = [set([1, 2, 3, 4, 5]), set([8, 9, 10, 11, 7]), set([6])]
    list_three_15 = [set([8, 9, 10, 7]), set([2, 3, 4, 5]), set([1]), set([6]), set([11])]
    list_four_15 = [set([2]), set([3]), set([7]), set([9]), set([6]), set([4]), set([1]), set([8]), set([10]), set([5]), set([11])]
    print("Example 3.1 ", autograderhw4.compute_q(graph3_15, list_one_15))
    print("Example 3.2 ", autograderhw4.compute_q(graph3_15, list_two_15))
    print("Example 3.3 ", autograderhw4.compute_q(graph3_15, list_three_15))
    print("Example 3.4 ", autograderhw4.compute_q(graph3_15, list_four_15))

def part_three_q2():
    '''
    Inputs: none
    Outputs: none
    
    Prints the resulting bar graph from running gn_graph_partitions on
    the karate club data in graph 3.13 from the book.
    '''
    figure_3_13 = bookgraphs.fig3_13g
    data = gn_graph_partition(figure_3_13)
    print(data)
    data_histogram = {}
    for item in data:
        data_histogram[len(item[1])] = item[0]
    plot_dist_linear(data_histogram, "Karate Club Data", "# of CCs", "Q value")

def part_three_q3():
    '''
    Inputs: none
    Outputs: none
    
    Prints the bar graph resulting from running gn_graph_partition on rice-facebook.repr
    '''
    facebook_graph = read_graph("/Users/abby/Desktop/Comp Projects/Comp 182/Homework 4/rice-facebook.repr")
    data = gn_graph_partition(facebook_graph)
    data_histogram = {}
    for item in data:
        data_histogram[len(item[1])] = item[0]
    plot_dist_linear(data_histogram, "Rice Facebook Subgraph Data", "# of CCs", "Q value")

def part_three_q3_attributes():
    '''
    Inputs: none
    Outputs: none
    
    Prints the grouped attributes from the group structure with the largest CC
    after running gn_graph_partition on rice-facebook.repr
    '''
    facebook_graph = read_graph("/Users/abby/Desktop/Comp Projects/Comp 182/Homework 4/rice-facebook.repr")
    data = gn_graph_partition(facebook_graph)
    CC = []
    max_q = 0
    for item in data:
        if item[0] > max_q:
            max_q = item[0]
            CC = item[1]
        if len(item[1])>20:
            break
    
    attributes = read_attributes("/Users/abby/Desktop/Comp Projects/Comp 182/Homework 4/rice-facebook-undergrads.txt")
    grouped_attributes = {}
    
    counter = 1
    for component in CC:
        for person in component:
            if counter in grouped_attributes:
                grouped_attributes[counter].append(attributes[person])
            else:
                grouped_attributes[counter] = [attributes[person]]
        counter += 1
    
    for key, value in grouped_attributes.items():
        print("CC ", key, value)

def highest_cc_attributes():
    '''
    Returns data from the highest Q grouping resulting from 
    running gn_graph_partition on facebook graph.
    '''
    CC = [[{'college': 1, 'year': 19, 'major': 34}, {'college': 1, 'year': 20, 'major': 1}, 
    {'college': 1, 'year': 20, 'major': 39}, {'college': 1, 'year': 20, 'major': 31}, {'college': 1, 'year': 21, 'major': 43}, 
    {'college': 1, 'year': 21, 'major': 23}, {'college': 1, 'year': 21, 'major': 26}, {'college': 1, 'year': 19, 'major': 21}, 
    {'college': 1, 'year': 20, 'major': 18}, {'college': 1, 'year': 19, 'major': 3}, {'college': 1, 'year': 20, 'major': 23}, 
    {'college': 1, 'year': 20, 'major': 22}, {'college': 1, 'year': 20, 'major': 22}, {'college': 1, 'year': 20, 'major': 1}, 
    {'college': 1, 'year': 21, 'major': 6}, {'college': 1, 'year': 21, 'major': 22}, {'college': 1, 'year': 21, 'major': 11}, 
    {'college': 1, 'year': 21, 'major': 34}, {'college': 1, 'year': 21, 'major': 55}, {'college': 1, 'year': 20, 'major': 34}, 
    {'college': 1, 'year': 21, 'major': 21}, {'college': 1, 'year': 20, 'major': 17}, {'college': 1, 'year': 21, 'major': 9}, 
    {'college': 1, 'year': 21, 'major': 22}, {'college': 1, 'year': 19, 'major': 21}, {'college': 1, 'year': 20, 'major': 15}, 
    {'college': 1, 'year': 20, 'major': 21}, {'college': 1, 'year': 21, 'major': 22}, {'college': 1, 'year': 20, 'major': 22}, 
    {'college': 1, 'year': 21, 'major': 18}, {'college': 1, 'year': 21, 'major': 48}, {'college': 1, 'year': 20, 'major': 35}, 
    {'college': 1, 'year': 21, 'major': 39}, {'college': 1, 'year': 21, 'major': 3}, {'college': 1, 'year': 21, 'major': 3}, 
    {'college': 1, 'year': 20, 'major': 9}, {'college': 1, 'year': 21, 'major': 17}, {'college': 1, 'year': 20, 'major': 17}, 
    {'college': 1, 'year': 19, 'major': 24}], [{'college': 2, 'year': 20, 'major': 17}, {'college': 2, 'year': 20, 'major': 25}, 
    {'college': 2, 'year': 21, 'major': 21}, {'college': 2, 'year': 20, 'major': 18}, {'college': 2, 'year': 19, 'major': 30}, 
    {'college': 2, 'year': 20, 'major': 30}, {'college': 2, 'year': 19, 'major': 12}, {'college': 2, 'year': 21, 'major': 53}, 
    {'college': 2, 'year': 21, 'major': 16}, {'college': 2, 'year': 20, 'major': 38}, {'college': 2, 'year': 20, 'major': 39}, 
    {'college': 2, 'year': 21, 'major': 32}, {'college': 2, 'year': 21, 'major': 24}, {'college': 2, 'year': 20, 'major': 3}, 
    {'college': 2, 'year': 19, 'major': 3}, {'college': 2, 'year': 20, 'major': 11}, {'college': 2, 'year': 20, 'major': 22}, 
    {'college': 2, 'year': 20, 'major': 1}, {'college': 2, 'year': 20, 'major': 22}, {'college': 2, 'year': 20, 'major': 26}, 
    {'college': 2, 'year': 20, 'major': 17}, {'college': 2, 'year': 21, 'major': 31}, {'college': 2, 'year': 21, 'major': 34}, 
    {'college': 2, 'year': 20, 'major': 18}, {'college': 2, 'year': 20, 'major': 22}, {'college': 2, 'year': 21, 'major': 21}, 
    {'college': 2, 'year': 20, 'major': 2}, {'college': 2, 'year': 20, 'major': 22}, {'college': 2, 'year': 21, 'major': 6}, 
    {'college': 2, 'year': 21, 'major': 3}, {'college': 2, 'year': 20, 'major': 38}, {'college': 2, 'year': 20, 'major': 31}, 
    {'college': 2, 'year': 21, 'major': 35}, {'college': 2, 'year': 20, 'major': 24}, {'college': 2, 'year': 20, 'major': 22}, 
    {'college': 2, 'year': 21, 'major': 9}, {'college': 2, 'year': 21, 'major': 23}, {'college': 2, 'year': 20, 'major': 35}, 
    {'college': 2, 'year': 21, 'major': 21}, {'college': 2, 'year': 20, 'major': 9}, {'college': 2, 'year': 21, 'major': 5}, 
    {'college': 2, 'year': 19, 'major': 34}, {'college': 2, 'year': 21, 'major': 8}, {'college': 2, 'year': 21, 'major': 6}, 
    {'college': 2, 'year': 20, 'major': 21}, {'college': 2, 'year': 21, 'major': 21}, {'college': 2, 'year': 20, 'major': 5}, 
    {'college': 2, 'year': 20, 'major': 31}], [{'college': 3, 'year': 21, 'major': 1}, {'college': 3, 'year': 20, 'major': 3}, 
    {'college': 3, 'year': 21, 'major': 33}, {'college': 3, 'year': 20, 'major': 30}, {'college': 3, 'year': 19, 'major': 11}, 
    {'college': 3, 'year': 19, 'major': 14}, {'college': 3, 'year': 21, 'major': 5}, {'college': 3, 'year': 21, 'major': 45}, 
    {'college': 3, 'year': 20, 'major': 30}, {'college': 3, 'year': 20, 'major': 6}, {'college': 3, 'year': 21, 'major': 21}, 
    {'college': 3, 'year': 21, 'major': 21}, {'college': 3, 'year': 21, 'major': 22}, {'college': 3, 'year': 20, 'major': 32}, 
    {'college': 3, 'year': 20, 'major': 5}, {'college': 3, 'year': 21, 'major': 55}, {'college': 3, 'year': 20, 'major': 45}, 
    {'college': 3, 'year': 20, 'major': 37}, {'college': 3, 'year': 20, 'major': 9}, {'college': 3, 'year': 20, 'major': 31}, 
    {'college': 3, 'year': 21, 'major': 25}, {'college': 3, 'year': 21, 'major': 23}, {'college': 3, 'year': 20, 'major': 15}, 
    {'college': 3, 'year': 21, 'major': 53}, {'college': 3, 'year': 21, 'major': 3}, {'college': 3, 'year': 21, 'major': 21}, 
    {'college': 3, 'year': 20, 'major': 21}, {'college': 3, 'year': 21, 'major': 20}, {'college': 3, 'year': 21, 'major': 15}, 
    {'college': 3, 'year': 21, 'major': 31}, {'college': 3, 'year': 21, 'major': 1}, {'college': 3, 'year': 21, 'major': 21}, 
    {'college': 3, 'year': 20, 'major': 18}, {'college': 3, 'year': 20, 'major': 21}, {'college': 3, 'year': 21, 'major': 34}, 
    {'college': 3, 'year': 20, 'major': 22}, {'college': 3, 'year': 21, 'major': 23}, {'college': 3, 'year': 20, 'major': 11}, 
    {'college': 3, 'year': 20, 'major': 29}, {'college': 3, 'year': 20, 'major': 1}, {'college': 3, 'year': 20, 'major': 31}],
    [{'college': 2, 'year': 21, 'major': 37}], [{'college': 1, 'year': 20, 'major': 39}], [{'college': 1, 'year': 21, 'major': 53}],
    [{'college': 3, 'year': 20, 'major': 9}], [{'college': 3, 'year': 19, 'major': 9}], [{'college': 3, 'year': 19, 'major': 30}],
    [{'college': 2, 'year': 20, 'major': 23}], [{'college': 3, 'year': 20, 'major': 7}], [{'college': 2, 'year': 21, 'major': 3}],
    [{'college': 1, 'year': 20, 'major': 6}], [{'college': 1, 'year': 20, 'major': 31}], [{'college': 3, 'year': 21, 'major': 11}],
    [{'college': 3, 'year': 21, 'major': 40}]]
    
    return CC

def analyze_attributes():
    counter = 1
    for component in highest_cc_attributes():
        print(counter, len(component))
        counter += 1

# part_three_q2()
# part_three_q3_attributes() -- this data went into highest_cc_attributes
# analyze_attributes()
# part_three_q3()
# test_part_3()
# graph3_20 = bookgraphs.fig3_18g 
# print("Shortest path edge betweenness: ", autograderhw4.shortest_path_edge_betweenness(graph3_20))