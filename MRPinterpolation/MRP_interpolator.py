import numpy as np
import networkx as nx

class MRP_interpolator:
    """
    Basic class implementing basic functions used by SD-MRP and WP-MRP.

    Attributes
    ----------
    original_grid : 2D numpy array
        the original grid supplied to be interpolated
    pred_grid : 2D numpy array
        interpolated version of original_grid
    G : networkx directed graph
        graph representation of pred_grid

    Methods
    -------
    reset():
        Returns pred_grid and G to their original state
        
    to_graph():
        Converts original_grid to a graph representation
        
    update_grid():
        Updates pred_grid to reflect changes to G
        
    get_pred_grid():
        Returns pred_grid
    """
    
    def __init__(self,grid):
        self.original_grid = grid.copy()
        self.pred_grid = grid.copy()
        self.G = self.to_graph()
        
        
    def __str__(self):
        return(str(self.pred_grid))
    
    
    def reset(self):
        """Return prediction grid and graph representation to their original form"""
        self.pred_grid = self.original_grid
        self.G = self.to_graph(self.original_grid)
        
        
    def to_graph(self,default_E=0):
        """
        Converts a grid to graph form.
        
        :param default_E: default expected value used for initialisation of the MRP
        """
        G = nx.DiGraph()
        
        grid_height = len(self.original_grid)
        grid_width = len(self.original_grid[0])
        
        for i in range(0,grid_height):
            for j in range(0,grid_width):
                val = self.original_grid[i][j]
                node_name = "r" + str(i) + "c" + str(j)
                G.add_node(node_name,y=val,E=default_E,r=i,c=j)

                # Connect to node above
                if(i > 0):
                    neighbour_name = "r" + str(i-1) + "c" + str(j)
                    G.add_edge(node_name,neighbour_name)
                    G.add_edge(neighbour_name,node_name)
                # Connect to node to the left
                if(j > 0):
                    neighbour_name = "r" + str(i) + "c" + str(j-1)
                    G.add_edge(node_name,neighbour_name)
                    G.add_edge(neighbour_name,node_name)

        return(G)


    def update_grid(self):
        """Update pred_grid to reflect changes to G"""
        for n in self.G.nodes(data=True):
            r = n[1]['r']
            c = n[1]['c']
            E = n[1]['E']
            self.pred_grid[r][c] = E
               
               
    def get_pred_grid(self):
        """Return pred_grid"""
        self.update_grid()
        return(self.pred_grid)