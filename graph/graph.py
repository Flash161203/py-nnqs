from typing import List

class Graph(object):
    """
    Base class for Graph.

    This class defines the structure or graph of the system.
    Graph must have an adjacency list.
    """

    def __init__(self):
        self.adj_list = None

    def set_adj_list(self, adj_list: List[List[int]]):
        self.adj_list = adj_list