import numpy as np
import networkx as nx

from .STMRP import STMRP

class SD_STMRP(STMRP):
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
        spatial discount parameter gamma used by SD-STMRP (typically 0-1
    tau : float
        temporal discount parameter tau used by SD-STMRP (typically 0-1)

    Methods
    -------
    run():
        Runs SD-MRP
        
    set_gamma():
        Sets gamma to a user-specified value
        
    set_tau():
        Sets tau to a user-specified value
        
    find_gamma():
        Automatically determines the best gamma (using subsampling or a training set)
    """
    
    def __init__(self,grid,auto_timestamps=False,gamma=0.9,tau=0.9):
        super(SD_STMRP, self).__init__(grid,auto_timestamps)
        self.gamma = gamma
        self.tau = tau
    
    def set_gamma(self,gamma):
        """
        Sets gamma to the manually supplied value.
        
        :param gamma: user-supplied gamma value
        """
        self.gamma = gamma
    
    def set_tau(self,tau):
        """
        Sets tau to the manually supplied value.
        
        :param tau: user-supplied tau value
        """
        self.tau = tau
            
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
                t = n[1]['t']
                for n1,n2,w in self.G.in_edges(n[0],data=True):
                    destination_node = self.G.nodes(data=True)[n1]
                    E_dest = destination_node['E']
                    t_dest = destination_node['t']
                    if(t_dest != t):
                        discount = self.tau
                    else:
                        discount = self.gamma
                    v_a = discount * E_dest
                    v_a_sum += v_a
                E_new = v_a_sum / len(self.G.in_edges(n[0]))
                nx.set_node_attributes(self.G,{n[0]:E_new},'E')

            else:
                nx.set_node_attributes(self.G,{n[0]:y},'E')
        
        self.update_grid()
        return(self.pred_grid)
        
    def find_discounts(self,search_epochs,subsample_proportion,iterations=100,ext=None):
        """
        Automatically sets gamma and tau to the best found value. Currently
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
            
            height = self.original_grid.shape[0]
            width = self.original_grid.shape[1]
            depth = self.original_grid.shape[2]
            
            sub_grid = self.original_grid.copy()
            for i in range(0,height):
                for j in range(0,width):
                    for t in range(0,depth):
                        if(not(np.isnan(sub_grid[i][j][t]))):
                            if(np.random.rand() < subsample_proportion):
                                sub_grid[i][j][t] = np.nan
                                       
            temp_MRP = SD_STMRP(sub_grid)
            
            best_loss = np.inf
            best_gamma = self.gamma
            best_tau = self.tau
            
            for ep in range(0,search_epochs):
                # Random search for best gamma for search_epochs iterations
                
                gamma = np.random.rand()
                tau = np.random.rand()
                temp_MRP.set_gamma(gamma)
                temp_MRP.set_tau(tau)
                pred_grid = temp_MRP.run(iterations)
                
                # Compute MAE of subsampled predictions
                err = 0
                err_count = 0
                               
                for i in range(0,height):
                    for j in range(0,width):
                        for t in range(0,depth):
                            if(not(np.isnan(self.original_grid[i][j][t]))):
                                if(np.isnan(sub_grid[i][j][t])):
                                    err += abs(pred_grid[i][j][t] - self.original_grid[i][j][t])
                                    err_count += 1
                mae = err / err_count
                if(mae < best_loss):
                    best_gamma = gamma
                    best_tau = tau
                    best_loss = mae
                
                temp_MRP.reset()
                
        self.gamma = best_gamma
        self.tau = best_tau
        
        return(best_gamma, best_tau)