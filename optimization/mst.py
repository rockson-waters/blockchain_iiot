import json
import sys
import numpy as np

from optimization.node_generator import get_latency, get_node_by_id

class Graph():
    INF = sys.maxsize
    def __init__(self, num_vertices):
        self.V = num_vertices
        self.parent = []
        self.graph = [[0 for column in range(num_vertices)] for row in range(num_vertices)]
        self.adj_matrix = np.asmatrix([0] * (num_vertices**2))
        self.adj_matrix.resize(num_vertices, num_vertices)

      
    # pretty print of the minimum spanning tree
    # prints the MST stored in the list var `parent`
    def printMST(self, parent):
        print("Edge     Weight")
        for i in range(1, self.V):
            print(f"{parent[i]} - {i}       {self.graph[i][parent[i]]}")

    # get adjacency graph
    def get_adjacency_matrix(self):
        """Creates adjacency matrix

        Returns:
            matrix: Adjacency matrix
        """
        for i in range(1, self.V):
            self.adj_matrix[self.parent[i], i] = 1
            self.adj_matrix[i, self.parent[i]] = 1
        return self.adj_matrix
  
    # finds the vertex with the minimum distance value
    # takes a key and the current set of nodes in the MST
    def minKey(self, key, mstSet):
        min = self.INF
        for v in range(self.V):
            if key[v] < min and mstSet[v] == False:
                min = key[v]
                min_index = v
        return min_index
  
    # prim's algo, graph is represented as an v by v adjacency list
    def prims(self):
        """Generates MST
        """
        # used to pick minimum weight edge
        key = [self.INF for _ in range(self.V)]
        # used to store MST
        parent = [None for _ in range(self.V)]
        # pick a random vertex, ie 0
        key[0] = 0
        # create list for t/f if a node is connected to the MST
        mstSet = [False for _ in range(self.V)]
        # set the first node to the root (ie have a parent of -1)
        parent[0] = -1

        for _ in range(self.V):
            # 1) pick the minimum distance vertex from the current key
            # from the set of points not yet in the MST
            u = self.minKey(key, mstSet)
            # 2) add the new vertex to the MST
            mstSet[u] = True

            # loop through the vertices to update the ones that are still
            # not in the MST
            for v in range(self.V):
                # if the edge from the newly added vertex (v) exists,
                # the vertex hasn't been added to the MST, and
                # the new vertex's distance to the graph is greater than the distance
                # stored in the initial graph, update the "key" value to the
                # distance initially given and update the parent of
                # of the vertex (v) to the newly added vertex (u)
                if self.graph[u][v] > 0 and mstSet[v] == False and key[v] > self.graph[u][v]:
                    key[v] = self.graph[u][v]
                    parent[v] = u
        
        self.parent = parent
        # self.printMST(parent)

    def save_solution(self, num_nodes=100, run_id=0):
        """Saves generated MST solution to file
        """
        neighbours = []
        solution = {}
        for i in range(0, self.V):
            
            neighbours = []
            for j in range(0, self.V):
                if self.adj_matrix[i,j] > 0:
                    neighbours.append(j)
            solution.update({i: neighbours})
        
          
        with open(f"out/{num_nodes}/{run_id}_mst_solution.json", 'w') as f:
            f.write(json.dumps(solution))

    def read_latencies(self):
        """Reads latencies from file and puts them in a matrix as weights for MST

        Returns:
            matrix: Adjency matrix
        """
        num_of_nodes = self.V
        a = [0] * (num_of_nodes**2)
        b = np.asmatrix(a)
        b = b.reshape(num_of_nodes, num_of_nodes)

        for i in range(0, num_of_nodes):
            node1 = get_node_by_id(i)
            for j in range(0, num_of_nodes):
                if (i == j):
                    b[i,j] = 0 #self.INF

                else:
                    node2 = get_node_by_id(j)
                    latency = get_latency(node1, node2)
                    b[i,j] = latency * 100
                    b[j,i] = latency * 100
        return b.tolist()



 
# g = Graph(5)
# g.graph = [ [0, 2, 0, 6, 0],
#            [2, 0, 3, 8, 5],
#            [0, 3, 0, 0, 7],
#            [6, 8, 0, 0, 9],
#            [0, 5, 7, 9, 0]]
 
# g.prims()
# g.get_adjacency_matrix()
# print("")
# print("")
# print(g.adj_matrix)
# g.save_solution()