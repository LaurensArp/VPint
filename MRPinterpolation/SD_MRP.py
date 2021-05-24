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
    
    
    
    def estimate_confidence(self):
        uncertainty_grid = self.original_grid.copy()
        uncertainty_grid = uncertainty_grid / uncertainty_grid      

        sub_MRP = SD_SMRP(uncertainty_grid)
        sub_MRP.find_gamma(100,0.5)
        confidence_pred_grid = sub_MRP.run(100)
        
        return(confidence_pred_grid)
            

    def run(self,iterations,confidence=False):
        """
        Runs SD-SMRP for the specified number of iterations. Creates a 3D (h,w,4) tensor val_grid, where the z-axis corresponds to a neighbour of each cell, and a 3D (h,w,4) weight tensor weight_grid, where the z-axis corresponds to the weights of every neighbour in val_grid's z-axis. The x and y axes of both tensors are stacked into 2D (h*w,4) matrices (one of which is transposed), after which the dot product is taken between both matrices, resulting in a (h*w,h*w) matrix. As we are only interested in multiplying the same row numbers with the same column numbers, we take the diagonal entries of the computed matrix to obtain a 1D (h*w) vector of updated values (we use numpy's einsum to do this efficiently, without wasting computation on extra dot products). This vector is then divided element-wise by a vector (flattened 2D grid) counting the number of neighbours of each cell, and we use the object's original_grid to replace wrongly updated known values to their original true values. We finally reshape this vector back to the original 2D pred_grid shape of (h,w).
        
        :param iterations: number of iterations used for the state value update function. If not specified, terminate once the maximal difference of a cell update dips below termination_threshold
        :returns: interpolated grid pred_grid
        """
        
        # Setup all this once
        
        height = self.pred_grid.shape[0]
        width = self.pred_grid.shape[1]
        
        h = height - 1
        w = width - 1
        
        neighbour_count_grid = np.zeros((height,width))
        weight_grid = np.zeros((height,width,4))
        global_weight_vec = np.ones(4) * self.gamma
        val_grid = np.zeros((height,width,4))
        
        # Compute weight grid once (vectorise at some point if possible)
        
        for i in range(0,height):
            for j in range(0,width):
                vec = np.ones(4) * self.gamma
                weight_grid[i,j,:] = vec
        
        weight_matrix = weight_grid.reshape((height*width,4)).transpose()
        
        # Set neighbour count and weight grids
        
        for i in range(0,height):
            for j in range(0,width):
                nc = 4
                if(i <= 0):
                    nc -= 1
                if(i >= height-1):
                    nc -= 1
                if(j <= 0):
                    nc -= 1
                if(j >= width-1):
                    nc -= 1
                neighbour_count_grid[i,j] = nc
        
        neighbour_count_vec = neighbour_count_grid.reshape(width*height)
        
        # Main loop
        
        for it in range(0,iterations):
            # Set val_grid
                     
            # Up
            val_grid[1:h+1,:,0] = self.pred_grid[0:h,:] # +1 because it's non-inclusive (0:10 means 0-9)
            val_grid[0,:,0] = np.zeros((width))
            
            # Right
            val_grid[:,0:w,1] = self.pred_grid[:,1:w+1]
            val_grid[:,w,1] = np.zeros((height))
            
            # Down
            val_grid[0:h,:,2] = self.pred_grid[1:h+1,:]
            val_grid[h,:,2] = np.zeros((width))
            
            # Left
            val_grid[:,1:w+1,3] = self.pred_grid[:,0:w]
            val_grid[:,0,3] = np.zeros((height))           
            
            # Compute new values, update pred grid
            
            val_matrix = val_grid.reshape((height*width,4)) # To do a dot product width weight matrix
            #new_grid = np.diag(np.dot(val_matrix,weight_matrix)) # Diag vector contains correct entries
            new_grid = np.einsum('ij,ji->i', val_matrix,weight_matrix) # Testing alternative to diag
            new_grid = new_grid / neighbour_count_vec # Correct for neighbour count
            flattened_original = self.original_grid.copy().reshape((height*width)) # can't use argwhere with 2D indexing
            new_grid[np.argwhere(~np.isnan(flattened_original))] = flattened_original[np.argwhere(~np.isnan(flattened_original))] # Keep known values from original
            
            new_grid = new_grid.reshape((height,width)) # Return to 2D grid
                     
            
            self.pred_grid = new_grid
            
        
        if(confidence):
            confidence_grid = self.estimate_confidence()
            return(self.pred_grid,confidence_grid)
        else:
            return(self.pred_grid)
        
    def run_old(self,iterations=None,termination_threshold=1e-4):
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
        
    def find_gamma(self,search_epochs,subsample_proportion,sub_iterations=100,ext=None):
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
                pred_grid = temp_MRP.run(sub_iterations)
                
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
    
    def __init__(self,grid,auto_timesteps=False,gamma=0.9,tau=0.9):
        super(SD_STMRP, self).__init__(grid,auto_timesteps)
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
        Runs SD-STMRP for the specified number of iterations. Creates a 4D (h,w,t,6) tensor val_grid, where the 4th axis corresponds to a neighbour of each cell, and a 4D (h,w,t,6) weight tensor weight_grid, where the 4th axis corresponds to the weights of every neighbour in val_grid's 4th axis. The x and y axes of both tensors are stacked into 2D (h*w*t,6) matrices (one of which is transposed), after which the dot product is taken between both matrices, resulting in a (h*w,h*w) matrix. As we are only interested in multiplying the same row numbers with the same column numbers, we take the diagonal entries of the computed matrix to obtain a 1D (h*w*t) vector of updated values (we use numpy's einsum to do this efficiently, without wasting computation on extra dot products). This vector is then divided element-wise by a vector (flattened 3D grid) counting the number of neighbours of each cell, and we use the object's original_grid to replace wrongly updated known values to their original true values. We finally reshape this vector back to the original 3D pred_grid shape of (h,w,t).
        
        :param iterations: number of iterations used for the state value update function. If not specified, terminate once the maximal difference of a cell update dips below termination_threshold
        :returns: interpolated grid pred_grid
        """
        
        # Setup all this once
        
        height = self.pred_grid.shape[0]
        width = self.pred_grid.shape[1]
        depth = self.pred_grid.shape[2]
        
        h = height - 1
        w = width - 1
        d = depth - 1
        
        neighbour_count_grid = np.zeros((height,width,depth))
        weight_grid = np.zeros((height,width,depth,6))
        val_grid = np.zeros((height,width,depth,6))
        
        # Compute weight grid once (vectorise at some point if possible)
        
        for i in range(0,height):
            for j in range(0,width):
                for t in range(0,depth):
                    vec = np.ones(6)
                    vec[0:4] = vec[0:4] * self.gamma
                    vec[4:6] = vec[4:6] * self.tau
                    weight_grid[i,j,t,:] = vec
        
        weight_matrix = weight_grid.reshape((height*width*depth,6)).transpose()
        
        # Set neighbour count and weight grids
        
        for i in range(0,height):
            for j in range(0,width):
                for t in range(0,depth):
                    nc = 6
                    if(i <= 0):
                        nc -= 1
                    if(i >= height-1):
                        nc -= 1
                    if(j <= 0):
                        nc -= 1
                    if(j >= width-1):
                        nc -= 1
                    if(t <= 0):
                        nc -= 1
                    if(t >= depth-1):
                        nc -= 1 
                    neighbour_count_grid[i,j,t] = nc

        
        neighbour_count_vec = neighbour_count_grid.reshape(width*height*depth) # TODO: verify that this is correct
        
        # Main loop
        
        for it in range(0,iterations):
            # Set val_grid
                      
            # Up
            val_grid[1:h+1,:,:,0] = self.pred_grid[0:h,:,:] # +1 because it's non-inclusive (0:10 means 0-9)
            val_grid[0,:,:,0] = np.zeros((width,depth))
            
            # Right
            val_grid[:,0:w,:,1] = self.pred_grid[:,1:w+1,:]
            val_grid[:,w,:,1] = np.zeros((height,depth))
            
            # Down
            val_grid[0:h,:,:,2] = self.pred_grid[1:h+1,:,:]
            val_grid[h,:,:,2] = np.zeros((width,depth))
            
            # Left
            val_grid[:,1:w+1,:,3] = self.pred_grid[:,0:w,:]
            val_grid[:,0,:,3] = np.zeros((height,depth)) 
            
            # Next time step
            val_grid[:,:,0:d,4] = self.pred_grid[:,:,1:d+1]
            val_grid[:,:,d,4] = np.zeros((height,width)) 
            
            # Previous time step
            val_grid[:,:,1:d+1,5] = self.pred_grid[:,:,0:d]
            val_grid[:,:,0,5] = np.zeros((height,width)) 
                      
            # Compute new values, update pred grid
            
            val_matrix = val_grid.reshape((height*width*depth,6)) # To do a dot product width weight matrix
            #new_grid = np.diag(np.dot(val_matrix,weight_matrix)) # Diag vector contains correct entries
            new_grid = np.einsum('ij,ji->i', val_matrix,weight_matrix)
            new_grid = new_grid / neighbour_count_vec # Correct for neighbour count
            flattened_original = self.original_grid.copy().reshape((height*width*depth)) # can't use argwhere with 3D indexing
            new_grid[np.argwhere(~np.isnan(flattened_original))] = flattened_original[np.argwhere(~np.isnan(flattened_original))] # Keep known values from original
            new_grid = new_grid.reshape((height,width,depth)) # Return to 3D grid          
            
            self.pred_grid = new_grid
        return(self.pred_grid)

        
    def find_discounts(self,search_epochs,subsample_proportion,sub_iterations=100,ext=None):
        """
        Automatically sets gamma and tau to the best found value. Currently
        only supports random search.
        
        :param search_epochs: number of epochs used by the random search
        :param subsample_proportion: proportion of training data used to compute errors
        :param iterations: number of MRP interations used by the random search
        :returns: best found value for gamma and tau
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
                pred_grid = temp_MRP.run(sub_iterations)
                
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