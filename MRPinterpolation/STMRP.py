import numpy as np
import networkx as nx
import datetime

class STMRP:
    """
    Basic class implementing the basic spatio-temporal framework for 
    SD-MRP and WP-MRP.

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
        
    dim_check():
        Checks the dimensions of supplied grid, transforms to
        3D grid if necessary
        
    set_timesteps():
        Automatically creates a 3D grid from a time-stamped
        dictionary of 2D spatial grids
        
    to_graph():
        Converts original_grid to a graph representation
        
    update_grid():
        Updates pred_grid to reflect changes to G
        
    get_pred_grid():
        Returns pred_grid
    """

    def __init__(self,data,auto_timesteps):       
        if(auto_timesteps):
            new_grid = self.set_timesteps(data.copy())
        else:
            new_grid = self.dim_check(data.copy())
            
        self.original_grid = new_grid
        self.pred_grid = new_grid
        self.G = self.to_graph(new_grid)
           
    def dim_check(self,grid):
        """
        Checks the dimensions of the supplied grid, and transforms
        it into a 3D grid (at a single time step) if necessary (this
        retains non-temporal spatial MRP functionality).
        
        :param grid: grid to check dimensions of
        :returns: suitable 3D grid
        """
        dims = grid.shape
        if(grid.ndim == 1):
            new_grid = np.zeros((dims[0],1,1))
            new_grid[:,0,0] = grid
        elif(grid.ndim == 2):
            new_grid = np.zeros((dims[0],dims[1],1))
            new_grid[:,:,0] = grid
        elif(grid.ndim == 3):
            new_grid = grid
        return(new_grid)
    
    
    def set_timesteps(self,data):
        """
        Checks the dimensions of the supplied grid, and transforms
        it into a 3D grid (at a single time step) if necessary (this
        retains non-temporal spatial MRP functionality).
        
        :param data: dictionary containing time stamps as keys and
        2D spatial grids as values. Time stamps MUST be chronologically
        ordered, and in string format "YYYY-MM-DD HH:MM:SS"
        :returns: spatio-temporal 3D grid
        """
        
        stamps = []
        min_stamp = None
        max_stamp = None
        min_gap = None
        i = 0
        for k,v in data.items():
            time_obj = datetime.datetime.strptime(k,"%Y-%m-%d %H:%M:%S")
            stamps.append(time_obj)
            if(i > 0):
                diff = time_obj - stamps[i-1]
                if(min_gap != None):
                    if(diff < min_gap):
                        min_gap = diff
                else:
                    min_gap = diff
                max_stamp = time_obj
                
            else:
                min_stamp = time_obj
                max_stamp = time_obj
            
            i += 1
        
        # Create empty 3D grid with (max-min)/gap time steps incremented by gap
        
        str_ind = min_stamp.strftime("%Y-%m-%d %H:%M:%S")
        height = data[str_ind].shape[0]
        width = data[str_ind].shape[1]
        depth = int((max_stamp - min_stamp) / min_gap) + 1
        
        grid = np.empty((height,width,depth))
        grid[:] = np.nan
        
        # Assign 2D grids to appropriate time steps
        
        for k,v in data.items():
            stamp = datetime.datetime.strptime(k,"%Y-%m-%d %H:%M:%S")
            ind = int((stamp - min_stamp) / min_gap)
            grid[:,:,ind] = v
        
        # Return 3D grid
        
        return(grid)
        
    def to_graph(self,grid,default_E=0):
        """
        Converts a grid to graph form.
        
        :param default_E: default expected value used for initialisation of the MRP
        """
        G = nx.DiGraph()
        
        grid_height = self.original_grid.shape[0]
        grid_width = self.original_grid.shape[1]
        grid_depth = self.original_grid.shape[2]
        
        for i in range(0,grid_height):
            for j in range(0,grid_width):
                for t in range(0,grid_depth):
                    val = grid[i][j][t]
                    node_name = "r" + str(i) + "c" + str(j) + "t" + str(t)
                    G.add_node(node_name,y=val,E=default_E,r=i,c=j,t=t)

                    # Connect to node above
                    if(i > 0):
                        neighbour_name = "r" + str(i-1) + "c" + str(j) + "t" + str(t)
                        G.add_edge(node_name,neighbour_name)
                        G.add_edge(neighbour_name,node_name)
                    # Connect to node to the left
                    if(j > 0):
                        neighbour_name = "r" + str(i) + "c" + str(j-1) + "t" + str(t)
                        G.add_edge(node_name,neighbour_name)
                        G.add_edge(neighbour_name,node_name)
                    # Connect to previous time step
                    # TODO: user-supplied neighbourhood
                    if(t > 0):
                        neighbour_name = "r" + str(i) + "c" + str(j) + "t" + str(t-1)
                        G.add_edge(node_name,neighbour_name)
                        G.add_edge(neighbour_name,node_name)

        return(G)
        
    def update_grid(self):
        """Update pred_grid to reflect changes to G"""
        for n in self.G.nodes(data=True):
            r = n[1]['r']
            c = n[1]['c']
            t = n[1]['t']
            E = n[1]['E']
            self.pred_grid[r][c][t] = E
               
    def get_pred_grid(self):
        """Return pred_grid"""
        self.update_grid()
        return(self.pred_grid)
    
    def reset(self):
        """Return prediction grid and graph representation to their original form"""
        self.pred_grid = self.original_grid.copy()
        self.G = self.to_graph(self.original_grid)
               
    def __str__(self):
        return(str(self.pred_grid))