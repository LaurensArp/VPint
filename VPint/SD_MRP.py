"""Module for MRP-based spatial interpolation.
"""

import numpy as np

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
    
    def __init__(self,grid,gamma=0.9,init_strategy='mean',mask=None):
        super().__init__(grid,init_strategy=init_strategy,mask=mask)
        self.gamma = gamma
    
    
    def set_gamma(self,gamma):
        """
        Sets gamma to the manually supplied value.
        
        :param gamma: user-supplied gamma value
        """
        self.gamma = gamma
            

    def run(self,iterations=-1,auto_terminate=True,auto_terminate_threshold=1e-4,track_delta=False,confidence=False):
        """
        Runs SD-SMRP for the specified number of iterations. Creates a 3D (h,w,4) tensor val_grid, where the z-axis corresponds to a neighbour of each cell, and a 3D (h,w,4) weight tensor weight_grid, where the z-axis corresponds to the weights of every neighbour in val_grid's z-axis. The x and y axes of both tensors are stacked into 2D (h*w,4) matrices (one of which is transposed), after which the dot product is taken between both matrices, resulting in a (h*w,h*w) matrix. As we are only interested in multiplying the same row numbers with the same column numbers, we take the diagonal entries of the computed matrix to obtain a 1D (h*w) vector of updated values (we use numpy's einsum to do this efficiently, without wasting computation on extra dot products). This vector is then divided element-wise by a vector (flattened 2D grid) counting the number of neighbours of each cell, and we use the object's original_grid to replace wrongly updated known values to their original true values. We finally reshape this vector back to the original 2D pred_grid shape of (h,w).
        
        :param iterations: number of iterations used for the state value update function. If not specified, default to 10000, which functions as the maximum number of iterations in case of non-convergence
        :param method: method for computing weights. Options: "predict" (using self.model), "cosine_similarity" (based on feature similarity), "exact" (compute average weight exactly for features)
        :param auto_terminate: if True, automatically terminate once the mean change in values after calling the update rule converges to a value under the auto_termination_threshold. Capped at 10000 iterations by default, though it usually takes under 100 iterations to converge
        :param auto_terminate_threshold: threshold for the amount of change as a proportion of the mean value of the grid, after which the algorithm automatically terminates
        :param track_delta: if True, return a vector containing the evolution of delta (mean proportion of change per iteration) along with the interpolated grid
        :returns: interpolated grid pred_grid
        """
        
        if(iterations > -1):
            auto_terminate = False
        else:
            iterations = 10000
        
        # Setup all this once
        
        height = self.pred_grid.shape[0]
        width = self.pred_grid.shape[1]
        
        h = height - 1
        w = width - 1
        
        neighbour_count_grid = np.zeros((height,width))
        #weight_grid = np.zeros((height,width,4))
        global_weight_vec = np.ones(4) * self.gamma
        val_grid = np.zeros((height,width,4))
        
        # Compute weight grid once (vectorise at some point if possible)
        
        #for i in range(0,height):
        #    for j in range(0,width):
        #        vec = np.ones(4) * self.gamma
        #        weight_grid[i,j,:] = vec
        weight_grid = np.ones((height,width,4)) * self.gamma
        
        weight_matrix = weight_grid.reshape((height*width,4)).transpose()
        
        # Set neighbour count grid
        
        neighbour_count_grid = np.ones(self.pred_grid.shape) * 4

        neighbour_count_grid[:,0] = neighbour_count_grid[:,0] - np.ones(neighbour_count_grid.shape[1])
        neighbour_count_grid[:,width-1] = neighbour_count_grid[:,width-1] - np.ones(neighbour_count_grid.shape[1])

        neighbour_count_grid[0,:] = neighbour_count_grid[0,:] - np.ones(neighbour_count_grid.shape[0])
        neighbour_count_grid[height-1,:] = neighbour_count_grid[height-1,:] - np.ones(neighbour_count_grid.shape[0])
        
        neighbour_count_vec = neighbour_count_grid.reshape(width*height)
        
        if(track_delta):
            delta_vec = np.zeros(iterations)
        
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
                     
            if(track_delta or auto_terminate):
                delta = np.nanmean(np.absolute(new_grid-self.pred_grid)) / np.nanmean(self.pred_grid)
                if(track_delta):
                    delta_vec[it] = delta
                if(auto_terminate):
                    if(delta <= auto_terminate_threshold):
                        self.pred_grid = new_grid
                        if(track_delta):
                            delta_vec = delta_vec[0:it+1]
                        break
            
            self.pred_grid = new_grid
            
        self.run_state = True
            
        if(track_delta):
            return(self.pred_grid,delta_vec)
        else:
            return(self.pred_grid)
            
        return(self.pred_grid)
        

        
    def find_gamma(self,search_epochs,subsample_proportion,sub_iterations=100,ext=None,max_gamma=1,min_gamma=0):
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
                gamma = np.random.uniform(low=min_gamma,high=max_gamma)
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
        
    def estimate_errors(self,hidden_prop=0.8):
        
        # Compute errors at subsampled known cells
        sub_grid = hide_values_uniform(self.original_grid.copy(),hidden_prop)
        sub_MRP = SD_SMRP(sub_grid)
        sub_MRP.set_gamma(self.gamma)
        sub_pred_grid = sub_MRP.run(100)
        err_grid = np.abs(self.original_grid.copy() - sub_pred_grid)

        # Predict errors for truly unknown cells
        sub_MRP = SD_SMRP(err_grid)
        err_gamma = sub_MRP.find_gamma(100,0.8,max_gamma=np.max(self.original_grid))
        err_grid_full = sub_MRP.run(100)
        
        return(err_grid_full)
        
        
        
class SD_STMRP(STMRP):
    """
    Class for SD-STMRP, extending STMRP

    Attributes
    ----------
    original_grid : 2D numpy array
        the original grid supplied to be interpolated
    pred_grid : 2D numpy array
        interpolated version of original_grid
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
        
        
    def run(self,iterations=-1,method='predict',auto_terminate=True,auto_terminate_threshold=1e-4,track_delta=False):
        """
        Runs SD-STMRP for the specified number of iterations. Creates a 4D (h,w,t,6) tensor val_grid, where the 4th axis corresponds to a neighbour of each cell, and a 4D (h,w,t,6) weight tensor weight_grid, where the 4th axis corresponds to the weights of every neighbour in val_grid's 4th axis. The x and y axes of both tensors are stacked into 2D (h*w*t,6) matrices (one of which is transposed), after which the dot product is taken between both matrices, resulting in a (h*w,h*w) matrix. As we are only interested in multiplying the same row numbers with the same column numbers, we take the diagonal entries of the computed matrix to obtain a 1D (h*w*t) vector of updated values (we use numpy's einsum to do this efficiently, without wasting computation on extra dot products). This vector is then divided element-wise by a vector (flattened 3D grid) counting the number of neighbours of each cell, and we use the object's original_grid to replace wrongly updated known values to their original true values. We finally reshape this vector back to the original 3D pred_grid shape of (h,w,t).
        
        :param iterations: number of iterations used for the state value update function. If not specified, terminate once the maximal difference of a cell update dips below termination_threshold
        :returns: interpolated grid pred_grid
        """
        
        # Setup all this once
        
        if(iterations > -1):
            auto_terminate = False
        else:
            iterations = 10000
        
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

        
        neighbour_count_vec = neighbour_count_grid.reshape(width*height*depth)
        
        if(track_delta):
            delta_vec = np.zeros(iterations)
        
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
            
            if(track_delta or auto_terminate):
                delta = np.nanmean(np.absolute(new_grid-self.pred_grid)) / np.nanmean(self.pred_grid)
                if(track_delta):
                    delta_vec[it] = delta
                if(auto_terminate):
                    if(delta <= auto_terminate_threshold):
                        self.pred_grid = new_grid
                        if(track_delta):
                            delta_vec = delta_vec[0:it+1]
                        break
            
            self.pred_grid = new_grid
            
        if(track_delta):
            return(self.pred_grid,delta_vec)
        else:
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