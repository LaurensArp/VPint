"""Module for MRP-based spatial interpolation.
"""

import numpy as np
import networkx as nx

from .MRP import SMRP, STMRP
        
        
class SD_SMRP(SMRP):
    """
    Class for SD-SMRP, extending SMRP

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
        
    set_gamma():
        Sets gamma to a user-specified value
        
    find_gamma():
        Automatically determines the best gamma (using subsampling or a training set)
        
    compute_confidence():
        compute an indication of uncertainty per pixel in pred_grid
    """
    
    def __init__(self,grid,gamma=0.9,init_strategy='zero'):
        super().__init__(grid,init_strategy=init_strategy)
        self.gamma = gamma
    
    
    def set_gamma(self,gamma):
        """
        Sets gamma to the manually supplied value.
        
        :param gamma: user-supplied gamma value
        """
        self.gamma = gamma
    
            
    def run(self,iterations=None,termination_threshold=1e-4):
        """
        Runs SD-SMRP for the specified number of iterations.
        
        :param iterations: optional number of iterations used for the state value update function. If not specified, terminate once the maximal difference of a cell update dips below termination_threshold
        :param termination_threshold: optional parameter specifying the threshold for auto-termination
        :returns: interpolated grid pred_grid
        """
        it = 0
        while True:
            delta = np.zeros(len(self.G.nodes))
            G = self.G.copy()
            c = 0
            
            # Iterate over nodes
            for n in self.G.nodes(data=True):
                r = n[1]['r']
                c = n[1]['c']
                y = n[1]['y']
                E = n[1]['E']

                # Interpolate missing nodes
                if(np.isnan(y)):
                    v_a_sum = 0
                    for n1,n2,w in self.G.in_edges(n[0],data=True):
                        destination_node = self.G.nodes(data=True)[n1]
                        E_dest = destination_node['E']
                        v_a = self.gamma * E_dest
                        v_a_sum += v_a
                    E_new = v_a_sum / len(self.G.in_edges(n[0]))
                    nx.set_node_attributes(G,{n[0]:E_new},'E')
                    
                    # Compute delta
                    delta[c] = abs(E - E_new)
                    c += 1
                else:
                    # Keep known values
                    nx.set_node_attributes(G,{n[0]:y},'E')
                    
                
            # Apply update
            self.G = G
            it += 1
            
            # Check termination conditions
            if(iterations != None):
                if(it >= iterations):
                    break
            else:
                if(np.max(delta) < termination_threshold):
                    break
            
        # Finalise
        self.update_grid()
        return(self.pred_grid)
    
    
    def run_experimental(self,iterations):
        height = self.pred_grid.shape[0]
        width = self.pred_grid.shape[1]
        
        for it in range(0,iterations):
            new_grid = self.pred_grid.copy()
            for i in range(0,height):
                for j in range(0,width):
                    if(np.isnan(self.original_grid[i,j])):
                        new_grid[i,j] = self.update_cell(new_grid,i,j)
        
            self.pred_grid = new_grid
        return(self.pred_grid)
                       
                    
    def update_cell(self,grid,i,j):
        conv_filter = np.array([
            [0, self.gamma, 0],
            [self.gamma, 0, self.gamma],
            [0, self.gamma, 0]
        ])
        
        r1_low = i-1
        r1_high = i+1
        c1_low = j-1
        c1_high = j-1
        
        r2_low = 0
        r2_high = 2
        c2_low = 0
        c2_high = 2
        
        if(i-1 < 0):
            r1_low = i
            r2_low = 1
        elif(i+1 > grid.shape[0]):
            r1_high = i
            r2_high = 1
        if(j-1 < 0):
            c1_low = j
            c2_low = 1
        elif(j+1 > grid.shape[1]):
            c1_high = j
            c2_low = 1
        
        vals = np.multiply(grid[r1_low:r1_high,c1_low:c1_high],conv_filter[r2_low:r2_high,c2_low:c2_high])
        return(np.sum(vals))
        
        
    def find_gamma(self,search_epochs,subsample_proportion,ext=None):
        """
        Automatically sets gamma to the best found value. Currently
        only supports random search.
        
        :param search_epochs: number of epochs used by the random search
        :param subsample_proportion: proportion of training data used to compute errors
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
                                                  
            best_loss = np.inf
            best_gamma = 0.9
            
            for ep in range(0,search_epochs):
                # Random search for best gamma for search_epochs iterations
                
                temp_MRP = SD_SMRP(sub_grid)
                gamma = np.random.rand()
                temp_MRP.set_gamma(gamma)
                pred_grid = temp_MRP.run()
                
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
        
    def compute_confidence(self,iterations=100):
        """
        Gives a confidence indication (float 0-1) for all cells in the grid by
        running a sub-MRP interpolation process on a grid where the confidence
        for known values is set to 1.
        
        :param iterations: number of iterations used for running the sub-MRP 
        :returns: confidence indication per pixel
        """
        height = self.original_grid.shape[0]
        width = self.original_grid.shape[1]
        new_grid = np.zeros((height,width)) + 1
        inds = np.isnan(self.original_grid)
        new_grid[inds] = np.nan
        
        temp_MRP = SD_SMRP(new_grid,gamma=self.gamma)
        confidence_grid = temp_MRP.run(iterations)
        return(confidence_grid)
        
        
        
class SD_STMRP(STMRP):
    """
    Class for SD-STMRP, extending STMRP

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
            
    def run(self,iterations=None,termination_threshold=1e-4):
        """
        Runs SD-STMRP for the specified number of iterations.
        
        :param iterations: number of iterations used for the state value update function
        :returns: interpolated grid pred_grid
        """
        it = 0
        while True:
            delta = np.zeros(len(self.G.nodes))
            G = self.G.copy()
            c = 0
            
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
                    nx.set_node_attributes(G,{n[0]:E_new},'E')
                    
                    # Compute delta
                    delta[c] = abs(E - E_new)
                    c += 1

                else:
                    nx.set_node_attributes(G,{n[0]:y},'E')
                
            # Apply update
            self.G = G
            it += 1
            
            # Check termination conditions
            if(iterations != None):
                if(it >= iterations):
                    break
            else:
                if(np.max(delta) < termination_threshold):
                    break
            
        # Finalise
        self.update_grid()
        return(self.pred_grid)
        
    def find_discounts(self,search_epochs,subsample_proportion,ext=None):
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
                pred_grid = temp_MRP.run()
                
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