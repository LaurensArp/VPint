import numpy as np
import networkx as nx

from MRP_interpolator import MRP_interpolator

class SD_MRP(MRP_interpolator):
    """
    Class for SD-MRP, extending MRP_interpolator

    Attributes
    ----------
    original_grid : 2D numpy array
        the original grid supplied to be interpolated
    pred_grid : 2D numpy array
        interpolated version of original_grid
    G : networkx directed graph
        graph representation of pred_grid
    gamma : float
        discount parameter gamma used by SD-MRP (typically 0-1)

    Methods
    -------
    run():
        Runs SD-MRP
        
    find_gamma():
        Automatically determines the best gamma (using subsampling or a training set)
    """
    
    def __init__(self,grid,gamma=0.9):
        self.original_grid = grid.copy()
        self.pred_grid = grid.copy()
        self.G = self.to_graph(grid)
        self.gamma = gamma
    
    
    def set_gamma(self,gamma):
        self.gamma = gamma
    
            
    def run(self,iterations):
        """
        Runs SD-MRP for the specified number of iterations.
        
        :param iterations: number of iterations used for the state value update function
        :returns: interpolated grid pred_grid
        """
    
        for n in self.G.nodes(data=True):
            r = n[1]['r']
            c = n[1]['c']
            y = n[1]['y']
            E = n[1]['E']
            
            if(np.isnan(y)):
                v_a_sum = 0
                for n1,n2,w in self.G.in_edges(n[0],data=True):
                    destination_node = self.G.nodes(data=True)[n1]
                    E_dest = destination_node['E']
                    v_a = self.gamma * E_dest
                    v_a_sum += v_a
                E_new = v_a_sum / len(self.G.in_edges(n[0]))
                nx.set_node_attributes(self.G,{n[0]:E_new},'E')

            else:
                nx.set_node_attributes(self.G,{n[0]:y},'E')
        
        self.update_grid()
        return(self.pred_grid)
        
        
    def find_gamma(self,search_epochs,subsample_proportion,iterations=100,ext=None):
        """
        Automatically sets gamma to the best found value. Currently
        only supports random search.
        
        :param search_epochs: number of epochs used by the random search
        :param subsample_proportion: proportion of training data used to compute errors
        :param iterations: number of MRP interations used by the random search
        :returns: best found value for gamma
        """
        
        if(ext != None):
            # Training set
            pass
        
        else:
            # Subsample
            
            sub_grid = self.original_grid.copy()
            for i in range(0,len(sub_grid)):
                for j in range(0,len(sub_grid[i])):
                    if(not(np.isnan(sub_grid[i][j]))):
                        if(np.random.rand() < subsample_proportion):
                            sub_grid[i][j] = np.nan
                                       
            temp_MRP = SD_MRP(sub_grid)
            
            best_loss = np.inf
            best_gamma = 0.9
            
            for ep in range(0,search_epochs):
                # Random search for best gamma for search_epochs iterations
                
                gamma = np.random.rand()
                temp_MRP.set_gamma(gamma)
                pred_grid = temp_MRP.run(iterations)
                
                # Compute MAE of subsampled predictions
                err = 0
                err_count = 0
                for i in range(0,len(self.original_grid)):
                    for j in range(0,len(self.original_grid[i])):
                        if(not(np.isnan(self.original_grid[i][j]))):
                            if(np.isnan(sub_grid[i][j])):
                                err += abs(pred_grid[i][j] - self.original_grid[i][j])
                                err_count += 1
                mae = err / err_count
                if(mae < best_loss):
                    best_gamma = gamma
                    best_loss = mae
                
                temp_MRP.reset()
                
        self.gamma = best_gamma
        return(best_gamma)