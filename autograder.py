## Abby Geigerman
## alg21
## COMP 182 Homework 4 autograder file

import random
from collections import *

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
    q = deque()

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

def compute_flow(g: dict, dist: dict, paths: dict) -> dict:
    '''
    Computes the flow across each edge E in a graph g. 
    Inputs:
        - g, a dictionary representing an adjaceny list
        - dist, a dictionary with the results of BFS from a start node
        - paths, a dictionary with the amount of shortest paths from 
        a start node to each node in g 

    Returns a dictionary mapping edges to the flow per edge. 
    '''
    # Initialize inflow and flow per edge dictionaries
    inflow = {}
    f = {}

    # Make list of unvisited nodes & sort them from farthest away to closest
    unvisited = list(g)
    unvisited = sorted(unvisited, key = lambda x: dist[x], reverse = True)

    # Add all edges to f and initialize all node inflows to 1
    for node in g:
        for nbr in g[node]:
            f[frozenset([node, nbr])] = 0
        inflow[node] = 1
        
    # Loop through all nodes and their neighbors, summing the inflows & calculating edge outflows
    for node in unvisited:
        for h in g[node]:
            if dist[h] < dist[node]:
                outflow = paths[h]/paths[node]*inflow[node] # calculation from book
                f[frozenset([node, h])] = outflow 
                inflow[h] += outflow # add edge outflow to nbr node 
    
    return f

def shortest_path_edge_betweenness(g: dict) -> dict:
    '''
    Computes the betweenness for each edge in graph g.
    Inputs:
        - g a dictionary representing an adjacency list
    
    Returns a dictionary with the keys being edges in g and the 
    values being the betweenness of that edge. 
    '''
    # Initialize f
    f = dict()
    
    # Map edges in f to value 0
    for node in g.keys():
        for nbr in g[node]:
            f[frozenset([node,nbr])] = 0

    # Run compute flow starting from each node
    for node in g:
        dist, npaths = bfs(g, node)
        flow = compute_flow(g, dist, npaths)

        for edge in flow: 
            f[edge] += flow[edge]

    return f

def compute_q(g: dict, c: list) -> float:
    # Initialize D
    D = [[0 for x in range(len(c))] for y in range(len(c))]

    # Calculate total edges
    total = sum(len(x) for x in g.values())
    
    # Initialize indices
    i = None
    j = None
    
    # Compute matrix D
    for node, nbrs in g.items():
        for nbr in nbrs:
            for index in range(len(c)):
                if node in c[index]:
                    i = index
                if nbr in c[index]:
                    j = index
            D[i][j] += 1
            if i != j:
                D[j][i] += 1
    # print(D)
    D = LinAl.multiply_by_val(D, 1/total)
    # print(D)

    # Compute value of Tr(D)
    Tr = 0
    for i in range(len(c)):
        Tr += D[i][i]
    # print(Tr)

    # Compute |D^2|
    D_squared = LinAl.multiply(D, D)
    D_squared_sum = LinAl.sum(D_squared)
    # print(D_squared_sum)

    # Calculate Q
    Q = Tr - D_squared_sum

    return Q

# Testing done in analysis
# dist, npaths = bfs(g, i)
# flow = compute_flow(g, dist, npaths)
# t = (1,2)
# print(tuple(reversed(t)))
