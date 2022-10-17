"""Module for MRP-based spatio-temporal interpolation.
"""

import numpy as np

from .MRP import SMRP, STMRP, VPintError
from .SD_MRP import SD_SMRP, SD_STMRP
from .utils.hide_spatial_data import hide_values_uniform

import math
        
        
class WP_SMRP(SMRP):
    """
    Class for WP-SMRP, extending SMRP.

    Attributes
    ----------
    original_grid : 2D numpy array
        the original grid supplied to be interpolated
    pred_grid : 2D numpy array
        interpolated version of original_grid
    feature_grid : 2D or 3D numpy array
        grid corresponding to original_grid, with feature vectors on the z-axis
    model : sklearn-based prediction model
        optional user-supplied machine learning model used to predict weights

    Methods
    -------
    run():
        Runs WP-MRP
        
    predict_weight():
        Predict the weight between two cells based on their features
        
    find_beta():
        Automatically determine the best value for beta
        
    contrast_map():
        Create a contrast map for a given input image (used in find_beta)
    
    get_weight_grid():
        Returns a grid of weights in a particular direction (can be useful for visualisations and debugging)
        
    get_weights():
        Returns the weights towards a particular cell based in its and its neighbours' features
        
    train():
        Train supplied prediction model on subsampled data or a training set
        
    compute_confidence():
        compute an indication of uncertainty per pixel in pred_grid
    """    
    
    def __init__(self,grid,feature_grid,model=None,init_strategy='mean',max_gamma=np.inf,min_gamma=0,mask=None):
        # Check shapes
        if(grid.shape[0] != feature_grid.shape[0] or grid.shape[1] != feature_grid.shape[1]):
            raise VPintError("Target and feature grids have different shapes: " + str(grid.shape) + " and " + str(feature_grid.shape))
        if(len(grid.shape)>2):
            # Reshape [None,None,1] to [None,None]
            if(grid.shape[2]==1):
                grid = grid[:,:,0]
            else:
                raise VPintError("Input grid needs to be two-dimensional, got " + str(grid.shape))
        super().__init__(grid,init_strategy=init_strategy,mask=mask)
        if(len(feature_grid.shape) == 3):
            self.feature_grid = feature_grid.copy().astype(float)
        elif(len(feature_grid.shape) == 2):
            self.feature_grid = feature_grid.copy().astype(float).reshape((feature_grid.shape[0],
                                                                          feature_grid.shape[1],
                                                                          1))
        else:
            raise VPintError("Improper feature dimensions; expected 2 or 3, got " + str(len(feature_grid.shape)))
        self.model = model 
        self.max_gamma = max_gamma
        self.min_gamma = min_gamma
        self._run_state = False
        self._run_method = "predict"
    
            
    def run(self,iterations=-1,method='exact',auto_terminate=True,auto_terminate_threshold=1e-4,track_delta=False, 
            confidence=False,confidence_model=None, save_gif=False,gif_path="convergence.gif", 
            auto_adapt=False,auto_adaptation_epochs=100,auto_adaptation_proportion=0.5, 
            auto_adaptation_strategy='random',auto_adaptation_max_iter=-1,
            auto_adaptation_subsample_strategy='max_contrast',
            auto_adaptation_verbose=False,
            prioritise_identity=False,priority_intensity=1, known_value_bias=0, 
            resistance=False,epsilon=0.01,mu=1.0):
        """
        Runs WP-SMRP for the specified number of iterations. Creates a 3D (h,w,4) tensor val_grid, where the z-axis corresponds to a neighbour of each cell, and a 3D (h,w,4) weight tensor weight_grid, where the z-axis corresponds to the weights of every neighbour in val_grid's z-axis. The x and y axes of both tensors are stacked into 2D (h*w,4) matrices (one of which is transposed), after which the dot product is taken between both matrices, resulting in a (h*w,h*w) matrix. As we are only interested in multiplying the same row numbers with the same column numbers, we take the diagonal entries of the computed matrix to obtain a 1D (h*w) vector of updated values (we use numpy's einsum to do this efficiently, without wasting computation on extra dot products). This vector is then divided element-wise by a vector (flattened 2D grid) counting the number of neighbours of each cell, and we use the object's original_grid to replace wrongly updated known values to their original true values. We finally reshape this vector back to the original 2D pred_grid shape of (h,w).
        
        :param iterations: number of iterations used for the state value update function. If not specified, default to 10000, which functions as the maximum number of iterations in case of non-convergence
        :param method: method for computing weights. Options: "predict" (using self.model), "cosine_similarity" (based on feature similarity), "exact" (compute average weight exactly for features)
        :param auto_terminate: if True, automatically terminate once the mean change in values after calling the update rule converges to a value under the auto_termination_threshold. Capped at 10000 iterations by default, though it usually takes under 100 iterations to converge
        :param auto_terminate_threshold: threshold for the amount of change as a proportion of the mean value of the grid, after which the algorithm automatically terminates (this is always relative/scaled to values)
        :param track_delta: if True, return a vector containing the evolution of delta (mean proportion of change per iteration) along with the interpolated grid
        :param confidence: return highly experimental confidence grid with predictions
        :param confidence_model: model used to create confidence grid
        :param save_gif: experimental code to save gif (not currently working)
        :param gif_path: file to save convergence gif to
        :param prioritise_identity: if True, predictions made using weights close to 1 will be weighted more heavily in the prediction dot product. For example, if a cell has 4 neighbours, 1 of which has a spatial weight of 1, and the others spatial weights of 0.1, 123 and 0.005, the predicted value will be largely based on the prediction resulting from the spatial weight of 1. If False, the predicted value will simply be the mean prediction of all four neighbours. Weights up to 1 are copied directly, and weights>1 are copied as 1/weight. This effect can be amplified by the priority_intensity parameter.
        :param priority_intensity: intensity of the identity prioritisation function. Set to 'auto' to automatically set this value using grid search on a subsampled proportion of the grid.
        :param auto_intensity_epochs: number of epochs for the random search if priority_intensity=='auto'
        :param auto_intensity_proportion: proportion of cells to subsample for the random search if priority_intensity=='auto'
        :param known_value_bias: if prioritise_identity==True, also give higher weight to predictions derived from cells (close to) known values. Determined using SD-MRP.
        :returns: interpolated grid pred_grid
        """      
        
        if(iterations > -1):
            auto_terminate = False
        else:
            iterations = 5000
               
        # Setup all this once
        
        height = self.pred_grid.shape[0]
        width = self.pred_grid.shape[1]
        
        h = height - 1
        w = width - 1
        
        neighbour_count_grid = np.zeros((height,width))
        weight_grid = np.zeros((height,width,4))
        val_grid = np.zeros((height,width,4))
        
        # Compute weight grid once (vectorise at some point if possible)
        
        # Ideally this if wouldn't be hardcodedhere , ML would also be vectorised, and predict_weight
        # would take entire grids as input
        if(method=='exact'):
            feature_size = self.feature_grid.shape[2]

            shp = self.feature_grid.shape
            size = np.product(shp)
            f_grid = self.feature_grid.reshape(size)
            f_grid[f_grid==0] = 0.01
            f_grid = f_grid.reshape(shp)

            # Every matrix contains feature vectors for the neighbour in some direction

            up_grid = np.ones((height,width,feature_size))
            right_grid = np.ones((height,width,feature_size))
            down_grid = np.ones((height,width,feature_size))
            left_grid = np.ones((height,width,feature_size))

            up_grid[1:-1,:,:] = f_grid[0:-2,:,:]
            right_grid[:,0:-2,:] = f_grid[:,1:-1,:]
            down_grid[0:-2,:,:] = f_grid[1:-1,:,:]
            left_grid[:,1:-1,:] = f_grid[:,0:-2,:]

            # Compute weights exacly

            up_weights = np.mean(f_grid / up_grid, axis=2)
            up_weights[0,:] = 0
            right_weights = np.mean(f_grid / right_grid, axis=2)
            right_weights[:,-1] = 0
            down_weights = np.mean(f_grid / down_grid, axis=2)
            down_weights[-1,:] = 0
            left_weights = np.mean(f_grid / left_grid, axis=2)
            left_weights[:,0] = 0

            weight_grid = np.stack([up_weights,right_weights,down_weights,left_weights],axis=-1)
            
            
        else:
            for i in range(0,height):
                for j in range(0,width):
                    vec = np.zeros(4)
                    f2 = self.feature_grid[i,j,:]
                    if(i > 0):
                        # Top
                        f1 = self.feature_grid[i-1,j,:]
                        vec[0] = self.predict_weight(f1,f2,method)
                    if(j < w):
                        # Right
                        f1 = self.feature_grid[i,j+1,:]
                        vec[1] = self.predict_weight(f1,f2,method)
                    if(i < h):
                        # Bottom
                        f1 = self.feature_grid[i+1,j,:]
                        vec[2] = self.predict_weight(f1,f2,method)
                    if(j > 0):
                        # Left
                        f1 = self.feature_grid[i,j-1,:]
                        vec[3] = self.predict_weight(f1,f2,method)

                    weight_grid[i,j,:] = vec
        
        weight_matrix = weight_grid.reshape((height*width,4)).transpose()
        
        # Set neighbour count grid
        
        neighbour_count_grid = np.ones(self.pred_grid.shape) * 4

        neighbour_count_grid[:,0] = neighbour_count_grid[:,0] - np.ones(neighbour_count_grid.shape[0])
        neighbour_count_grid[:,width-1] = neighbour_count_grid[:,width-1] - np.ones(neighbour_count_grid.shape[0])

        neighbour_count_grid[0,:] = neighbour_count_grid[0,:] - np.ones(neighbour_count_grid.shape[1])
        neighbour_count_grid[height-1,:] = neighbour_count_grid[height-1,:] - np.ones(neighbour_count_grid.shape[1])
        
        neighbour_count_vec = neighbour_count_grid.reshape(width*height)
        
        # Search for best parameters
        if(auto_adapt):
            params = []
            if(prioritise_identity):
                params.append('beta')
            if(resistance):
                params.append('epsilon')
                params.append('mu')
                
            params_opt = self.auto_adapt(params, auto_adaptation_epochs,auto_adaptation_proportion,
                                                          search_strategy=auto_adaptation_strategy, 
                                                          max_sub_iter=auto_adaptation_max_iter,
                                                          subsample_strategy=auto_adaptation_subsample_strategy)
            if(auto_adaptation_verbose):
                print("Best found params: " + str(params_opt))
            for k,v in params_opt.items():
                if(k=='beta'):
                    priority_intensity = params_opt[k]
                elif(k=='epsilon'):
                    epsilon = params_opt[k]
                elif(k=='mu'):
                    mu = params_opt[k]
                
        if(priority_intensity==0):
            prioritise_identity=False
        if(prioritise_identity):
            # Prioritise weights close to 1, under the assumption they will be
            # more informative/constant. 
            
            if(priority_intensity==0):
                # Same as no priority
                priority_grid = np.ones(weight_grid.shape)
            else:
                priority_grid = weight_grid.copy()
                priority_grid[priority_grid>1] = 1/priority_grid[priority_grid>1] * (1/priority_grid[priority_grid>1] / (1/priority_grid[priority_grid>1] * priority_intensity))
                priority_grid[priority_grid<1] = priority_grid[priority_grid<1] * (priority_grid[priority_grid<1] / ((priority_grid[priority_grid<1]+0.001) * priority_intensity))
            
            # Prioritise known values
            if(known_value_bias>0):
                priority_grid2 = np.zeros(priority_grid.shape)
                for d in range(0,priority_grid.shape[2]):
                    priority_grid2[:,:,d] = self.original_grid.copy()
                priority_grid2[~np.isnan(priority_grid2)] = 1
                priority_grid3 = priority_grid2.copy()
                for d in range(0,priority_grid3.shape[2]):
                    sub_MRP = SD_SMRP(priority_grid2[:,:,d],init_strategy='zero',gamma=(1-known_value_bias))
                    priority_grid3[:,:,d] = sub_MRP.run()
                priority_grid = np.multiply(priority_grid,priority_grid3)
                # Smooth for 0
                priority_vec = priority_grid.reshape(np.product(priority_grid.shape))
                priority_vec[priority_vec==0] = 0.01
                priority_grid = priority_vec.reshape(priority_grid.shape)
        
        # Track amount of change over iterations
        if(track_delta):
            delta_vec = np.zeros(iterations)
            
        # Compute mean value of available values (used for penalisation of strong deviations)
        target_mean = np.nanmean(self.original_grid)
        
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
            if(prioritise_identity):
                # element wise multiplication weight+vals
                individual_predictions = np.multiply(weight_matrix.transpose(),val_matrix)
                # dot product with priority weights
                new_grid = np.einsum('ij,ji->i', individual_predictions,priority_grid.reshape(height*width,4).transpose()) 
                # divide by sum of priority weights
                new_grid = new_grid / np.sum(priority_grid,axis=2).reshape(height*width)
                
            else:   
                new_grid = np.einsum('ij,ji->i', val_matrix,weight_matrix)
                new_grid = new_grid / neighbour_count_vec # Correct for neighbour count
                
            flattened_original = self.original_grid.copy().reshape((height*width)) # can't use argwhere with 2D indexing
            new_grid[np.argwhere(~np.isnan(flattened_original))] = flattened_original[np.argwhere(~np.isnan(flattened_original))] # Keep known values from original           
            new_grid = new_grid.reshape((height,width)) # Return to 2D grid
            
            # Apply 'resistance' to make it harder to deviate further from the mean
            if(resistance):
               
                # Add 'elastic band resistance'
                # Up until threshold (band length), increase as much as desired (band is slack)
                # After threshold, add resistance scaled by how far away from threshold we are
                
                # Flatten arrays
                shp = new_grid.shape
                size = np.product(shp)
                new_grid_vec = new_grid.reshape(size)
                pred_grid_vec = self.pred_grid.reshape(size)
                
                # Compute delta_y for all pixels
                delta_y = new_grid_vec - pred_grid_vec
                
                # Update pixels < threshold as old + delta # TODO: check if just doing all is faster
                inds = np.where(new_grid_vec<=mu)
                new_grid_vec[inds] = pred_grid_vec[inds] + delta_y[inds]
                
                # Update pixels > threshold as old + delta - k*old (or old+delta?)
                inds = np.where(new_grid_vec>mu)
                new_grid_vec[inds] = pred_grid_vec[inds] + delta_y[inds] - (epsilon*pred_grid_vec[inds])
            
            # Compute delta, save to vector and/or auto-terminate where relevant
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
        self.run_method = method
                        
        if(track_delta):
            return(self.pred_grid,delta_vec)
        else:
            return(self.pred_grid)
        
        
    def predict_weight(self,f1,f2,method):
        """
        Predict the weight between two cells based on feature vectors f1 and f2, using either self.model, cosine similarity or the average exact weight computed from the two feature vectors.
        
        :param f1: feature vector for the neighbouring cell
        :param f2: feature vector for the cell of interest
        :param method: method for computing weights. Options: "predict" (using self.model), "cosine_similarity" (based on feature similarity), "exact" (compute average weight exactly for features)
        :returns: predicted weight gamma
        """
        if(method == "predict"):
            f = np.concatenate((f1,f2))
            f = f.reshape(1,len(f))
            # TODO: better solution
            gamma = 1.0
            try:
                gamma = self.model.predict(f)[0]
            except:
                gamma = 1.0
        elif(method == "cosine_similarity"):
            gamma = np.dot(f1,f2) / max(np.sum(f1) * np.sum(f2),0.01)
        elif(method == "exact"):
            f1_temp = f1.copy()
            f1_temp[f1_temp == 0] = 0.01
            gamma = np.mean(f2 / f1_temp) 
        elif(method == "exact_inverse"):
            f2_temp = f2.copy()
            f2_temp[f2_temp == 0] = 0.01
            gamma = np.mean(f1 / f2_temp) 
        else:
            raise VPintError("Invalid weight prediction method: " + str(method))
        gamma = max(self.min_gamma,min(gamma,self.max_gamma))
        return(gamma)
    
    def auto_adapt(self,params,search_epochs,subsample_proportion,search_strategy='random',subsample_strategy='max_contrast',ranges={},max_sub_iter=-1,hill_climbing_threshold=5):
        """
        Automatically sets the identity priority intensity parameter to the best found value. Currently
        only supports random search.
        
        :param params: dict of parameters to search for (supported: beta, epsilon, mu)
        :param search_epochs: number of epochs used by the random search
        :param subsample_proportion: proportion of training data used to compute errors
        :returns: best found value for identity priority intensity
        """         

        # Subsample
        if(subsample_strategy=='random'):
            sub_grid = self.original_grid.copy()

            shp = sub_grid.shape
            size = np.product(shp)

            rand_vec = np.random.rand(size)
            sub_vec = sub_grid.reshape(size)

            sub_vec[rand_vec<subsample_proportion] = np.nan
            sub_grid = sub_vec.reshape(shp)


        elif(subsample_strategy=='max_diff'):
            sub_grid = self.original_grid.copy()
            diff = np.absolute(self.original_grid - np.mean(self.feature_grid,axis=2))
            
            shp = sub_grid.shape
            size = np.product(shp)

            diff_vec = diff.reshape(size)
            sub_vec = sub_grid.reshape(size)
            
            # Get indices of sorted array
            num_pixels = int(subsample_proportion * len(sub_vec[~np.isnan(sub_vec)]))
            diff_vec = np.nan_to_num(diff_vec,nan=-1.0)
            temp = np.argpartition(-diff_vec,num_pixels)
            result_args = temp[:num_pixels]
            
            # Replace most different pixels by nan
            sub_vec[result_args] = np.nan
            sub_grid = sub_vec.reshape(shp)
            
        elif(subsample_strategy=='max_contrast'):
            sub_grid = self.original_grid.copy()
            contrast_grid = self.contrast_map(sub_grid)
            
            shp = sub_grid.shape
            size = np.product(shp)

            contrast_vec = contrast_grid.reshape(size)
            sub_vec = sub_grid.reshape(size)
            
            # Get indices of sorted array
            num_pixels = int(subsample_proportion * len(sub_vec[~np.isnan(sub_vec)]))
            contrast_vec = np.nan_to_num(contrast_vec,nan=-1.0)
            temp = np.argpartition(-contrast_vec,num_pixels)
            result_args = temp[:num_pixels]
            
            # Replace most different pixels by nan
            sub_vec[result_args] = np.nan
            sub_grid = sub_vec.reshape(shp)
            
        else:
            raise VPintError("Invalid subsample strategy: " + str(subsample_strategy))
            
        # Set bounds/distribution if specified, default otherwise
        bounds = {'beta':(0,4),'epsilon':(0.1,0.01),
                  'mu':(np.nanmean(self.original_grid),3*np.nanmean(self.original_grid))}
        for k,v in ranges:
            bounds[k] = v
            
        # Initialise params, best stuff
        best_loss = np.inf
        best_val = {}
        for k in params:
            if(k not in ['beta','epsilon','mu']):
                raise VPintError("Invalid parameter to optimise: " + str(k))
            best_val[k] = -1
        
        if(search_strategy=='random'):
            ran_default = False
            for ep in range(0,search_epochs):
                # Random search for best val for search_epochs iterations

                temp_MRP = WP_SMRP(sub_grid,self.feature_grid.copy())
                
                # Make sure to check defaults first, explore randomly otherwise
                if(not(ran_default)):
                    vals = {}
                    for k in params:
                        if(k=='beta'):
                            vals[k] = 1
                        if(k=='epsilon'):
                            vals[k] = 0
                        if(k=='mu'):
                            vals[k] = np.nanmean(self.original_grid) + 2*np.nanstd(self.original_grid)
                    ran_default = True
                else:
                    vals = {}
                    for k in params:
                        if(k=='beta'):
                            vals[k] = np.random.randint(low=bounds[k][0],high=bounds[k][1])
                        if(k=='epsilon'):
                            # Technically not min/max, but mean/std. Bounded by 0-1
                            #vals[k] = min(max(0,np.random.normal(bounds[k][0],bounds[k][1])),1)
                            vals[k] = min(max(0,np.random.uniform(low=0,high=0.3)),1)
                        if(k=='mu'):
                            vals[k] = np.random.uniform(low=bounds[k][0],high=bounds[k][1])

                # Kind of hacky, for different combinations of parameters
                # Currently mu can only be optimised if epsilon is too
                if('beta' in vals and 'epsilon' in vals):
                    if('mu' in vals):
                        mu = vals['mu']
                    else:
                        mu = np.nanmean(self.original_grid) + 2*np.nanstd(self.original_grid)
                    pred_grid = temp_MRP.run(prioritise_identity=True, priority_intensity=vals['beta'], 
                                             iterations=max_sub_iter, auto_adapt=False,
                                             resistance=True, epsilon=vals['epsilon'],mu=mu)
                    
                elif('beta' in vals and not('epsilon' in vals)):
                    pred_grid = temp_MRP.run(prioritise_identity=True, priority_intensity=vals['beta'], 
                                             iterations=max_sub_iter, auto_adapt=False)
                    
                elif(not('beta' in vals) and 'epsilon' in vals):
                    if('mu' in vals):
                        mu = vals['mu']
                    else:
                        mu = np.nanmean(self.original_grid) + 2*np.nanstd(self.original_grid)
                    pred_grid = temp_MRP.run(resistance=True, epsilon=vals['epsilon'],mu=mu, auto_adapt=False)

                # Compute MAE of subsampled predictions
                mae = np.nanmean(np.absolute(pred_grid.reshape(np.product(pred_grid.shape)) - self.original_grid.reshape(np.product(self.original_grid.shape))))

                # Update where necessary
                if(mae < best_loss):
                    best_vals = vals
                    best_loss = mae

                temp_MRP.reset()
                
        elif(search_strategy=='sequential'):
            if('beta' not in params and 'epsilon' not in params):
                raise VPintError("Cannot use sequential auto-adaptation without both beta and epsilon/mu")
                
            best_vals = {}
                
            # Grid search for beta        

            for val in range(0,8):
                # Grid search

                temp_MRP = WP_SMRP(sub_grid,self.feature_grid.copy())
                pred_grid = temp_MRP.run(prioritise_identity=True,priority_intensity=val,iterations=max_sub_iter, 
                                        auto_adapt=False)

                # Compute MAE of subsampled predictions
                mae = np.nanmean(np.absolute(pred_grid.reshape(np.product(pred_grid.shape)) - self.original_grid.reshape(np.product(self.original_grid.shape))))

                if(mae < best_loss):
                    best_val = val
                    best_loss = mae

                temp_MRP.reset()
                
            best_beta = best_val
            params.pop(params.index('beta'))
            
            # Random search for epsilon/mu
            
            ran_default = False
            for ep in range(0,search_epochs):
                # Random search for best val for search_epochs iterations

                temp_MRP = WP_SMRP(sub_grid,self.feature_grid.copy())
                
                # Make sure to check defaults first, explore randomly otherwise
                if(not(ran_default)):
                    vals = {}
                    for k in params:
                        if(k=='epsilon'):
                            vals[k] = 0
                        if(k=='mu'):
                            vals[k] = np.nanmean(self.original_grid) + 2*np.nanstd(self.original_grid)
                    ran_default = True
                else:
                    vals = {}
                    for k in params:
                        if(k=='epsilon'):
                            # Technically not min/max, but mean/std. Bounded by 0-1
                            #vals[k] = min(max(0,np.random.normal(bounds[k][0],bounds[k][1])),1)
                            vals[k] = min(max(0,np.random.uniform(low=0,high=0.5)),1)
                        if(k=='mu'):
                            vals[k] = np.random.uniform(low=bounds[k][0],high=bounds[k][1])

                # Kind of hacky, for different combinations of parameters
                # Currently mu can only be optimised if epsilon is too

                if('mu' in vals):
                    mu = vals['mu']
                else:
                    mu = np.nanmean(self.original_grid) + 2*np.nanstd(self.original_grid)
                pred_grid = temp_MRP.run(resistance=True, epsilon=vals['epsilon'],mu=mu, auto_adapt=False, 
                                        prioritise_identity=True, priority_intensity=best_beta)

                # Compute MAE of subsampled predictions
                mae = np.nanmean(np.absolute(pred_grid.reshape(np.product(pred_grid.shape)) - self.original_grid.reshape(np.product(self.original_grid.shape))))

                # Update where necessary
                if(mae < best_loss):
                    best_vals = vals
                    best_loss = mae

                temp_MRP.reset()
                
            best_vals['beta'] = best_beta
                
                
        elif(search_strategy=='hill_climbing'):
            
            ran_default = False
            static_counter = 0
            for ep in range(0,search_epochs):
                # Random search for best val for search_epochs iterations

                temp_MRP = WP_SMRP(sub_grid,self.feature_grid.copy())
                
                # Make sure to check defaults first
                if(not(ran_default)):
                    vals = {}
                    for k in params:
                        if(k=='beta'):
                            vals[k] = 1
                        if(k=='epsilon'):
                            vals[k] = 0
                        if(k=='mu'):
                            vals[k] = np.nanmean(self.original_grid) + 2*np.nanstd(self.original_grid)
                else:
                    for k in params:
                        if(k=='beta'):
                            r = np.random.rand()
                            if(r <= 0.33):
                                vals[k] = max(bounds[k][0], vals[k]-1)
                            elif(r > 0.33 and r <= 0.66):
                                vals[k] = min(bounds[k][1], vals[k]+1)
                            else:
                                pass
                        if(k=='epsilon'):
                            r = np.random.rand()
                            # Epsilon bounds are mean/std, can be used for steps
                            if(r <= 0.33):
                                vals[k] = max(0, vals[k]-bounds[k][1])
                            elif(r > 0.33 and r <= 0.66):
                                vals[k] = min(1, vals[k]+bounds[k][1])
                            else:
                                pass
                        if(k=='mu'):
                            factor = np.random.uniform(low=0,high=np.nanstd(self.original_grid))
                            if(r <= 0.33):
                                vals[k] = max(bounds[k][0], vals[k]-factor)
                            elif(r > 0.33 and r <= 0.66):
                                vals[k] = min(bounds[k][1], vals[k]+factor)
                            else:
                                pass

                # Kind of hacky, for different combinations of parameters
                # Currently mu can only be optimised if epsilon is too
                if('beta' in vals and 'epsilon' in vals):
                    if('mu' in vals):
                        mu = vals['mu']
                    else:
                        mu = np.nanmean(self.original_grid) + 2*np.nanstd(self.original_grid)
                    pred_grid = temp_MRP.run(prioritise_identity=True, priority_intensity=vals['beta'], 
                                             iterations=max_sub_iter, auto_adapt=False,
                                             resistance=True, epsilon=vals['epsilon'],mu=mu)
                    
                elif('beta' in vals and not('epsilon' in vals)):
                    pred_grid = temp_MRP.run(prioritise_identity=True, priority_intensity=vals['beta'], 
                                             iterations=max_sub_iter, auto_adapt=False)
                    
                elif(not('beta' in vals) and 'epsilon' in vals):
                    if('mu' in vals):
                        mu = vals['mu']
                    else:
                        mu = np.nanmean(self.original_grid) + 2*np.nanstd(self.original_grid)
                    pred_grid = temp_MRP.run(resistance=True, epsilon=vals['epsilon'],mu=mu, auto_adapt=False)

                # Compute MAE of subsampled predictions
                mae = np.nanmean(np.absolute(pred_grid.reshape(np.product(pred_grid.shape)) - self.original_grid.reshape(np.product(self.original_grid.shape))))

                # Update where necessary
                if(mae < best_loss):
                    best_vals = vals
                    best_loss = mae
                    static_counter = 0
                else:
                    # Early stopping if no more improvement
                    static_counter += 1
                    if(static_counter >= hill_climbing_threshold):
                        #print("Early stopping: " + str(ep+1)) # TODO: remove
                        break
                        
                # Set to exploration defaults where appropriate
                if(not(ran_default)):
                    vals = {}
                    for k in params:
                        if(k=='beta'):
                            vals[k] = 1
                        if(k=='epsilon'):
                            vals[k] = 0.1
                        if(k=='mu'):
                            vals[k] = np.nanmean(self.original_grid) + 2*np.nanstd(self.original_grid)
                    ran_default = True
                    
                temp_MRP.reset()
                

        else:
            raise VPintError("Invalid search strategy: " + str(search_strategy))
        
        for k,v in best_vals.items():
            if(k in params and v==-1):
                print("WARNING: no " + k + " better than dummy, please check your code (defaulting to 1)")
                best_vals[k] = 1
            if(not(k in params)):
                best_vals.pop(k) # Remove parameters that were not optimised
            
        return(best_vals)
    
    def find_beta_old(self,search_epochs,subsample_proportion,search_strategy='grid',subsample_strategy='max_contrast',min_val=0,max_val=10,max_sub_iter=-1):
        """
        Automatically sets the identity priority intensity parameter to the best found value. Currently
        only supports random search.
        
        :param search_epochs: number of epochs used by the random search
        :param subsample_proportion: proportion of training data used to compute errors
        :returns: best found value for identity priority intensity
        """

        # Subsample
        if(subsample_strategy=='random'):
            sub_grid = self.original_grid.copy()

            shp = sub_grid.shape
            size = np.product(shp)

            rand_vec = np.random.rand(size)
            sub_vec = sub_grid.reshape(size)

            sub_vec[rand_vec<subsample_proportion] = np.nan
            sub_grid = sub_vec.reshape(shp)


        elif(subsample_strategy=='max_diff'):
            sub_grid = self.original_grid.copy()
            diff = np.absolute(self.original_grid - np.mean(self.feature_grid,axis=2))
            
            shp = sub_grid.shape
            size = np.product(shp)

            diff_vec = diff.reshape(size)
            sub_vec = sub_grid.reshape(size)
            
            # Get indices of sorted array
            num_pixels = int(subsample_proportion * len(sub_vec[~np.isnan(sub_vec)]))
            diff_vec = np.nan_to_num(diff_vec,nan=-1.0)
            temp = np.argpartition(-diff_vec,num_pixels)
            result_args = temp[:num_pixels]
            
            # Replace most different pixels by nan
            sub_vec[result_args] = np.nan
            sub_grid = sub_vec.reshape(shp)
            
        elif(subsample_strategy=='max_contrast'):
            sub_grid = self.original_grid.copy()
            contrast_grid = self.contrast_map(sub_grid)
            
            shp = sub_grid.shape
            size = np.product(shp)

            contrast_vec = contrast_grid.reshape(size)
            sub_vec = sub_grid.reshape(size)
            
            # Get indices of sorted array
            num_pixels = int(subsample_proportion * len(sub_vec[~np.isnan(sub_vec)]))
            contrast_vec = np.nan_to_num(contrast_vec,nan=-1.0)
            temp = np.argpartition(-contrast_vec,num_pixels)
            result_args = temp[:num_pixels]
            
            # Replace most different pixels by nan
            sub_vec[result_args] = np.nan
            sub_grid = sub_vec.reshape(shp)
            
        else:
            raise VPintError("Invalid subsample strategy: " + str(subsample_strategy))
            
        best_loss = np.inf
        best_val = -1
        
        if(search_strategy=='random'):
            for ep in range(0,search_epochs):
                # Random search for best val for search_epochs iterations

                temp_MRP = WP_SMRP(sub_grid,self.feature_grid.copy())
                if(best_val==-1):
                    val = 1 # try default first
                else:
                    val = np.random.randint(low=min_val,high=max_val)
                pred_grid = temp_MRP.run(prioritise_identity=True,priority_intensity=val,iterations=max_sub_iter)

                # Compute MAE of subsampled predictions
                mae = np.nanmean(np.absolute(pred_grid.reshape(np.product(pred_grid.shape)) - self.original_grid.reshape(np.product(self.original_grid.shape))))

                if(mae < best_loss):
                    best_val = val
                    best_loss = mae

                temp_MRP.reset()
                
        elif(search_strategy=='grid'):
            val = 1 # try default first
            temp_MRP = WP_SMRP(sub_grid,self.feature_grid.copy())
            pred_grid = temp_MRP.run(prioritise_identity=True,priority_intensity=val,iterations=max_sub_iter)

            # Compute MAE of subsampled predictions
            mae = np.nanmean(np.absolute(pred_grid.reshape(np.product(pred_grid.shape)) - self.original_grid.reshape(np.product(self.original_grid.shape))))

            if(mae < best_loss):
                best_val = val
                best_loss = mae

            temp_MRP.reset()
                
            for val in range(min_val,max_val):
                # Random search for best val for search_epochs iterations

                temp_MRP = WP_SMRP(sub_grid,self.feature_grid.copy())
                pred_grid = temp_MRP.run(prioritise_identity=True,priority_intensity=val,iterations=max_sub_iter)

                # Compute MAE of subsampled predictions
                mae = np.nanmean(np.absolute(pred_grid.reshape(np.product(pred_grid.shape)) - self.original_grid.reshape(np.product(self.original_grid.shape))))

                if(mae < best_loss):
                    best_val = val
                    best_loss = mae

                temp_MRP.reset()
                
        else:
            raise VPintError("Invalid search strategy: " + str(search_strategy))
                
        if(best_val==-1):
            print("WARNING: no identity priority intensity better than dummy, please check your code (defaulting to 1)")
            best_val = 1
            
        return(best_val)
        
    
    def contrast_map(self,grid):
        """
        Create a contrast map of the feature grid, which can be used by find_beta to select pixels to sample. Contrast is computed as the mean average distance between a pixel and its neighbours, normalised to a 0-1 range.
        
        :param grid: input grid to create a contrast map for
        :returns: contrast map
        """
        height = grid.shape[0]
        width = grid.shape[1]
        
        # Create neighbour count grid
        neighbour_count_grid = np.ones(grid.shape) * 4

        neighbour_count_grid[:,0] = neighbour_count_grid[:,0] - np.ones(neighbour_count_grid.shape[1])
        neighbour_count_grid[:,width-1] = neighbour_count_grid[:,width-1] - np.ones(neighbour_count_grid.shape[1])

        neighbour_count_grid[0,:] = neighbour_count_grid[0,:] - np.ones(neighbour_count_grid.shape[0])
        neighbour_count_grid[height-1,:] = neighbour_count_grid[height-1,:] - np.ones(neighbour_count_grid.shape[0])

        # Create (h*w*4) value grid
        val_grid = np.zeros((height,width,4))
        
        up_grid = np.zeros((height,width))
        right_grid = np.zeros((height,width))
        down_grid = np.zeros((height,width))
        left_grid = np.zeros((height,width))

        up_grid[1:-1,:] = grid[0:-2,:]
        right_grid[:,0:-2] = grid[:,1:-1]
        down_grid[0:-2,:] = grid[1:-1,:]
        left_grid[:,1:-1] = grid[:,0:-2]
        
        val_grid[:,:,0] = up_grid
        val_grid[:,:,1] = right_grid
        val_grid[:,:,2] = down_grid
        val_grid[:,:,3] = left_grid
        
        # Compute contrast as average absolute distance
        temp_grid = np.repeat(grid[:,:,np.newaxis],4,axis=2)
        diff = np.absolute(val_grid-temp_grid)
        sum_diff = np.nansum(diff,axis=-1)
        avg_contrast = sum_diff / neighbour_count_grid
        
        min_val = np.nanmin(avg_contrast)
        max_val = np.nanmax(avg_contrast)
        avg_contrast = np.clip((avg_contrast-min_val)/(max_val-min_val), 0,1)
        
        return(avg_contrast)
    
    
    def get_weight_grid(self,method='exact',direction='up'):
        """
        Simple function that can be used to visualise weights. Return a grid where every cell corresponds to the weight it receives from the cell in a given direction.
        
        :param method: prediction method, as used in run()
        :param direction: direction from which the weights are visualised
        :returns: weight grid
        """
        # Setup all this once
        
        height = self.pred_grid.shape[0]
        width = self.pred_grid.shape[1]
        
        h = height - 1
        w = width - 1
        
        weight_grid = np.zeros((height,width,4))
        
        # Compute weight grid once (vectorise at some point if possible)
        
        
        for i in range(0,height):
            for j in range(0,width):
                vec = np.zeros(4)
                f2 = self.feature_grid[i,j,:]
                if(i > 0):
                    # Top
                    f1 = self.feature_grid[i-1,j,:]
                    vec[0] = self.predict_weight(f1,f2,method)
                if(j < w):
                    # Right
                    f1 = self.feature_grid[i,j+1,:]
                    vec[1] = self.predict_weight(f1,f2,method)
                if(i < h):
                    # Bottom
                    f1 = self.feature_grid[i+1,j,:]
                    vec[2] = self.predict_weight(f1,f2,method)
                if(j > 0):
                    # Left
                    f1 = self.feature_grid[i,j-1,:]
                    vec[3] = self.predict_weight(f1,f2,method)

                weight_grid[i,j,:] = vec
        
        # Note: this is incoming, so this is the weight from above
        if(direction=='up'):
            return(weight_grid[:,:,0])
        elif(direction=='right'):
            return(weight_grid[:,:,1])
        elif(direction=='down'):
            return(weight_grid[:,:,2])
        elif(direction=='left'):
            return(weight_grid[:,:,3])
        else:
            raise VPintError("Invalid direction: " + str(direction))
    
    def get_weights(self,i,j,method="predict"):
        weights = {}
        
        # Up
        if(i > 0):
            f2 = self.feature_grid[i,j,:]
            f1 = self.feature_grid[i-1,j,:]
            gamma = self.predict_weight(f1,f2,method)
            weights['up'] = gamma
        else:
            weights['up'] = np.nan
        
        # Right
        if(j < self.feature_grid.shape[1]-1):
            f2 = self.feature_grid[i,j,:]
            f1 = self.feature_grid[i,j+1,:]
            gamma = self.predict_weight(f1,f2,method)
            weights['right'] = gamma
        else:
            weights['right'] = np.nan
            
        # Down
        if(i < self.feature_grid.shape[0]-1):
            f2 = self.feature_grid[i,j,:]
            f1 = self.feature_grid[i+1,j,:]
            gamma = self.predict_weight(f1,f2,method)
            weights['down'] = gamma
        else:
            weights['down'] = np.nan   
        
        # Left
        if(j > 0):
            f2 = self.feature_grid[i,j,:]
            f1 = self.feature_grid[i,j-1,:]
            gamma = self.predict_weight(f1,f2,method)
            weights['left'] = gamma
        else:
            weights['left'] = np.nan
       
        return(weights)
        
        
    def train(self,training_set=None,training_features=None,limit_training=True):
        
        if(training_set is None or training_features is None):
            training_set = self.original_grid.copy()
            training_features = self.feature_grid.copy()
        
        # Get training size
        
        height = training_set.shape[0]
        width = training_set.shape[1]
        
        h = height - 1
        w = width - 1

        c = 0
        for i in range(0,height):
            for j in range(0,width):
                y2 = training_set[i,j]
                if(not(np.isnan(y2))):
                    if(i > 0):
                        # Top
                        y1 = training_set[i,j]
                        if(not(np.isnan(y1))):
                            c += 1
                    if(j < w):
                        # Right
                        y1 = training_set[i,j+1]
                        if(not(np.isnan(y1))):
                            c += 1                        
                    if(i < h):
                        # Bottom                       
                        y1 = training_set[i+1,j]
                        if(not(np.isnan(y1))):
                            c += 1  
                    if(j > 0):
                        # Left                       
                        y1 = training_set[i,j-1]
                        if(not(np.isnan(y1))):
                            c += 1
        
        training_size = c
        if(training_size == 0):
            return(False) # Allow for downstream handling of empty training sets
        num_features = training_features.shape[2] * 2
        
        X_train = np.zeros((training_size,num_features))
        y_train = np.zeros(training_size)
        
        c = 0
        for i in range(0,height):
            for j in range(0,width):
                f2 = training_features[i,j,:]
                y2 = training_set[i,j]
                if(not(np.isnan(y2))):
                    if(i > 0):
                        # Top
                        y1 = training_set[i,j]
                        if(not(np.isnan(y1))):
                            f1 = training_features[i-1,j,:]
                            f = np.concatenate((f1,f2))
                            f = f.reshape(1,len(f))
                            gamma = y2 / max(0.01,y1)
                            if(limit_training):
                                gamma = max(self.min_gamma,min(gamma,self.max_gamma))
                            X_train[c,:] = f
                            y_train[c] = gamma
                            c += 1
                    if(j < w):
                        # Right
                        y1 = training_set[i,j+1]
                        if(not(np.isnan(y1))):
                            f1 = training_features[i,j+1,:]
                            f = np.concatenate((f1,f2))
                            f = f.reshape(1,len(f))
                            gamma = y2 / max(0.01,y1)
                            if(limit_training):
                                gamma = max(self.min_gamma,min(gamma,self.max_gamma))
                            X_train[c,:] = f
                            y_train[c] = gamma
                            c += 1                        
                    if(i < h):
                        # Bottom                       
                        y1 = training_set[i+1,j]
                        if(not(np.isnan(y1))):
                            f1 = training_features[i+1,j,:]
                            f = np.concatenate((f1,f2))
                            f = f.reshape(1,len(f))
                            gamma = y2 / max(0.01,y1)
                            if(limit_training):
                                gamma = max(self.min_gamma,min(gamma,self.max_gamma))
                            X_train[c,:] = f
                            y_train[c] = gamma
                            c += 1  
                    if(j > 0):
                        # Left                       
                        y1 = training_set[i,j-1]
                        if(not(np.isnan(y1))):
                            f1 = training_features[i,j-1,:]
                            f = np.concatenate((f1,f2))
                            f = f.reshape(1,len(f))
                            gamma = y2 / max(0.01,y1)
                            if(limit_training):
                                gamma = max(self.min_gamma,min(gamma,self.max_gamma))
                            X_train[c,:] = f
                            y_train[c] = gamma
                            c += 1
                            
                            
        aug = np.append(X_train,y_train.reshape((len(y_train),1)),axis=1)
        aug = aug[~np.isnan(aug).any(axis=1),:]
        
        X_train = aug[:,0:-1]
        y_train = aug[:,-1]
        
        self.model.fit(X_train,y_train)
        
    def estimate_errors(self,hidden_prop=0.8,method='exact'):
        
        # Compute errors at subsampled known cells
        sub_grid = hide_values_uniform(self.original_grid.copy(),hidden_prop)
        sub_MRP = WP_SMRP(sub_grid,self.feature_grid.copy(),None)
        sub_pred_grid = sub_MRP.run(100,method=method)
        err_grid = np.abs(self.original_grid.copy() - sub_pred_grid)
        # TODO replace known values

        # Predict errors for truly unknown cells
        sub_MRP = SD_SMRP(err_grid)
        err_gamma = sub_MRP.find_gamma(100,hidden_prop)
        err_grid_full = sub_MRP.run(100)
        
        return(err_grid_full)
    
    def confidence_map(self,hidden_prop=0.8,interp_method="SD_MRP"):
        
        # Compute errors at subsampled known cells
        sub_grid = hide_values_uniform(self.original_grid.copy(),hidden_prop)
        sub_MRP = WP_SMRP(sub_grid,self.feature_grid.copy())
        sub_pred_grid = sub_MRP.run()
        err_grid = np.absolute(self.original_grid.copy() - sub_pred_grid)
        
        # Normalise to 0-1 confidence range
        min_val = np.nanmin(err_grid)
        max_val = np.nanmax(err_grid)
        t1 = (err_grid-min_val)
        t2 = (max_val-min_val)
        conf_grid = np.clip(t1/t2,0,1)
        conf_grid = 1 - conf_grid

        # Predict errors for truly unknown cells
        if(interp_method=="SD_MRP"):
            sub_MRP = SD_SMRP(conf_grid,init_strategy="zero")
            conf_gamma = sub_MRP.find_gamma(100,hidden_prop)
            conf_grid_final = sub_MRP.run(100)
        elif(interp_method=="WP_MRP"):
            from sklearn.linear_model import LinearRegression
            sub_MRP = WP_SMRP(conf_grid,self.feature_grid.copy(),model=LinearRegression())
            sub_MRP.train()
            conf_grid_final = sub_MRP.run(method="predict")
        
        return(conf_grid_final)
    
    def confidence_map2(self,hidden_prop=0.8,interp_method="WP_MRP",smooth=True,kernel_size=3,smooth_iterations=1):
        
        # Compute errors at subsampled known cells
        sub_grid = hide_values_uniform(self.original_grid.copy(),hidden_prop)
        sub_MRP = WP_SMRP(sub_grid,self.feature_grid.copy())
        sub_pred_grid = sub_MRP.run()
        err_grid = np.absolute(self.original_grid.copy() - sub_pred_grid)
        
        # Normalise to 0-1 confidence range
        min_val = np.nanmin(err_grid)
        max_val = np.nanmax(err_grid)
        t1 = (err_grid-min_val)
        t2 = (max_val-min_val)
        conf_grid = np.clip(t1/t2,0,1)
        conf_grid = 1 - conf_grid

        # Predict errors for truly unknown cells
        if(interp_method=="SD_MRP"):
            sub_MRP = SD_SMRP(conf_grid,init_strategy="zero")
            conf_gamma = sub_MRP.find_gamma(100,hidden_prop)
            conf_grid_final = sub_MRP.run(100)
        elif(interp_method=="WP_MRP"):
            try:
                import cv2
            except:
                raise VPintError("Failed to import cv2; please install it to use this functionality")
            contr_map = self.contrast_map(self.feature_grid.copy()[:,:,0])
            diff_map = np.absolute(self.feature_grid[:,:,0] - self.original_grid)
            #diff_map = 1 - np.clip((diff_map-np.nanmin(diff_map))/(np.nanmax(diff_map)-np.nanmin(diff_map)),0,1)
            sub_MRP = WP_SMRP(diff_map,contr_map)
            conf_grid_final = sub_MRP.run()
            
            conf_grid_final = 1 - np.clip((conf_grid_final-np.nanmin(conf_grid_final))/(np.nanmax(conf_grid_final)-np.nanmin(conf_grid_final)),0,1)
            #conf_grid_final[conf_grid_final>1] = 1
            
            if(smooth):
                kernel = np.ones((kernel_size,kernel_size), np.float32)/(kernel_size*kernel_size)
                for i in range(smooth_iterations):
                    conf_grid_final = cv2.filter2D(src=conf_grid_final, ddepth=-1, kernel=kernel)
            
        
        sz = np.product(conf_grid.shape)
        shp = conf_grid.shape
        conf_grid_final_vec = conf_grid_final.reshape(sz)
        mask_vec = self.original_grid.copy().reshape(sz)
        conf_grid_final_vec[~np.isnan(mask_vec)] = 1
        conf_grid_final = conf_grid_final_vec.reshape(shp)
        return(conf_grid_final)
         
               
        
class WP_STMRP(STMRP):
    """
    Class for WP-STMRP, extending STMRP

    Attributes
    ----------
    original_grid : 3D numpy array
        the original grid supplied to be interpolated
    pred_grid : 3D numpy array
        interpolated version of original_grid
    feature_grid : 4D numpy array
        grid corresponding to original_grid, with feature vectors on the z-axis
    model : sklearn-based prediction model
        user-supplied machine learning model used to predict weights

    Methods
    -------
    run():
        Runs WP-MRP
        
    train():
        Train supplied prediction model on subsampled data or a training set
    """    
    def __init__(self,grid,feature_grid,model_spatial=None,model_temporal=None,auto_timesteps=False,max_gamma=np.inf,min_gamma=0):
        # Feature grid is a 3d grid, where x and y correspond to grid, and the z axis contains feature
        # vectors
        
        super(WP_STMRP, self).__init__(grid,auto_timesteps)
        self.feature_grid = feature_grid.copy().astype(float)
        self.model_spatial = model_spatial
        self.model_temporal = model_temporal
        self.max_gamma = max_gamma
        self.min_gamma = min_gamma
        
        
    def run(self,iterations=-1,method='predict',auto_terminate=True,auto_terminate_threshold=1e-4,track_delta=False):
        """
        Runs WP-STMRP for the specified number of iterations. Creates a 4D (h,w,t,6) tensor val_grid, where the 4th axis corresponds to a neighbour of each cell, and a 4D (h,w,t,6) weight tensor weight_grid, where the 4th axis corresponds to the weights of every neighbour in val_grid's 4th axis. The x and y axes of both tensors are stacked into 2D (h*w*t,6) matrices (one of which is transposed), after which the dot product is taken between both matrices, resulting in a (h*w,h*w) matrix. As we are only interested in multiplying the same row numbers with the same column numbers, we take the diagonal entries of the computed matrix to obtain a 1D (h*w*t) vector of updated values (we use numpy's einsum to do this efficiently, without wasting computation on extra dot products). This vector is then divided element-wise by a vector (flattened 3D grid) counting the number of neighbours of each cell, and we use the object's original_grid to replace wrongly updated known values to their original true values. We finally reshape this vector back to the original 3D pred_grid shape of (h,w,t).
        
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
                    vec = np.zeros(6)
                    f2 = self.feature_grid[i,j,:]
                    if(i > 0):
                        # Top
                        f1 = self.feature_grid[i-1,j,:]
                        f3 = np.array([0])
                        vec[0] = self.predict_weight(f1,f2,f3,method)
                    if(j < w):
                        # Right
                        f1 = self.feature_grid[i,j+1,:]
                        f3 = np.array([0])
                        vec[1] = self.predict_weight(f1,f2,f3,method)
                    if(i < h):
                        # Bottom
                        f1 = self.feature_grid[i+1,j,:]
                        f3 = np.array([0])
                        vec[2] = self.predict_weight(f1,f2,f3,method)
                    if(j > 0):
                        # Left
                        f1 = self.feature_grid[i,j-1,:]
                        f3 = np.array([0])
                        vec[3] = self.predict_weight(f1,f2,f3,method)
                    if(t > 0):
                        # Before
                        f1 = self.feature_grid[i,j,:]
                        f3 = np.array([-1])
                        vec[4] = self.predict_weight(f1,f2,f3,method)
                    if(t < d):
                        # After
                        f1 = self.feature_grid[i,j,:]
                        f3 = np.array([1])
                        vec[5] = self.predict_weight(f1,f2,f3,method)

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
            new_grid = new_grid.reshape((height,width,depth)) # Return to 2D grid
            
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
    
    
    def predict_weight(self,f1,f2,f3,method):
        """
        Predict the weight between two cells based on feature vectors f1 and f2, using either self.model, cosine similarity or the average exact weight computed from the two feature vectors.
        
        :param f1: feature vector for the neighbouring cell
        :param f2: feature vector for the cell of interest
        :param method: method for computing weights. Options: "predict" (using self.model), "cosine_similarity" (based on feature similarity), "exact" (compute average weight exactly for features)
        :returns: predicted weight gamma
        """
        if(method == "predict"):
            f = np.concatenate((f1,f2,f3))
            f = f.reshape(1,len(f))
            if(f[0,-1] == -1 or f[0,-1] == 1):
                gamma = self.model_temporal.predict(f)[0]
            else:
                gamma = self.model_spatial.predict(f)[0]
            gamma = max(self.min_gamma,min(gamma,self.max_gamma))
        elif(method == "cosine_similarity"):
            gamma = np.dot(f1,f2) / max(np.sum(f1) * np.sum(f2),0.01)
        elif(method == "exact"):
            f1_temp = f1.copy()
            f1_temp[f1_temp == 0] = 0.01
            gamma = np.mean(f2 / f1_temp) 
        else:
            raise VPintError("Invalid method: " + method)
        return(gamma)
    
            
    
    def train(self):
        # Get training size
        
        height = self.pred_grid.shape[0]
        width = self.pred_grid.shape[1]
        depth = self.pred_grid.shape[2]
        
        h = height - 1
        w = width - 1
        d = depth - 1

        c_spatial = 0
        c_temporal = 0
        for i in range(0,height):
            for j in range(0,width):
                for t in range(0,depth):
                    y2 = self.original_grid[i,j,t]
                    if(not(np.isnan(y2))):
                        if(i > 0):
                            # Top
                            y1 = self.original_grid[i,j,t]
                            if(not(np.isnan(y1))):
                                c_spatial += 1
                        if(j < w):
                            # Right
                            y1 = self.original_grid[i,j+1,t]
                            if(not(np.isnan(y1))):
                                c_spatial += 1                        
                        if(i < h):
                            # Bottom                       
                            y1 = self.original_grid[i+1,j,t]
                            if(not(np.isnan(y1))):
                                c_spatial += 1  
                        if(j > 0):
                            # Left                       
                            y1 = self.original_grid[i,j-1,t]
                            if(not(np.isnan(y1))):
                                c_spatial += 1
                        if(t > 0):
                            # Before
                            y1 = self.original_grid[i,j,t-1]
                            if(not(np.isnan(y1))):
                                c_temporal += 1
                        if(t < d):
                            # After                       
                            y1 = self.original_grid[i,j,t+1]
                            if(not(np.isnan(y1))):
                                c_temporal += 1
        
        training_size_spatial = c_spatial
        training_size_temporal = c_temporal
        num_features = self.feature_grid.shape[2] * 2 + 1
        
        X_train_spatial = np.zeros((training_size_spatial,num_features))
        y_train_spatial = np.zeros(training_size_spatial)
        
        X_train_temporal = np.zeros((training_size_temporal,num_features))
        y_train_temporal = np.zeros(training_size_temporal)
        
        c_spatial = 0
        c_temporal = 0
        for i in range(0,height):
            for j in range(0,width):
                for t in range(0,depth):
                    f2 = self.feature_grid[i,j,:]
                    y2 = self.original_grid[i,j,t]
                    if(not(np.isnan(y2))):
                        if(i > 0):
                            # Top
                            y1 = self.original_grid[i,j,t]
                            if(not(np.isnan(y1))):
                                f1 = self.feature_grid[i-1,j,:]
                                f3 = np.array([0])
                                f = np.concatenate((f1,f2,f3))
                                f = f.reshape(1,len(f))
                                gamma = y2 / max(0.01,y1)
                                X_train_spatial[c_spatial,:] = f
                                y_train_spatial[c_spatial] = gamma
                                c_spatial += 1
                        if(j < w):
                            # Right
                            y1 = self.original_grid[i,j+1,t]
                            if(not(np.isnan(y1))):
                                f1 = self.feature_grid[i,j+1,:]
                                f3 = np.array([0])
                                f = np.concatenate((f1,f2,f3))
                                f = f.reshape(1,len(f))
                                gamma = y2 / max(0.01,y1)
                                X_train_spatial[c_spatial,:] = f
                                y_train_spatial[c_spatial] = gamma
                                c_spatial += 1                        
                        if(i < h):
                            # Bottom                       
                            y1 = self.original_grid[i+1,j,t]
                            if(not(np.isnan(y1))):
                                f1 = self.feature_grid[i+1,j,:]
                                f3 = np.array([0])
                                f = np.concatenate((f1,f2,f3))
                                f = f.reshape(1,len(f))
                                gamma = y2 / max(0.01,y1)
                                X_train_spatial[c_spatial,:] = f
                                y_train_spatial[c_spatial] = gamma
                                c_spatial += 1  
                        if(j > 0):
                            # Left                       
                            y1 = self.original_grid[i,j-1,t]
                            if(not(np.isnan(y1))):
                                f1 = self.feature_grid[i,j-1,:]
                                f3 = np.array([0])
                                f = np.concatenate((f1,f2,f3))
                                f = f.reshape(1,len(f))
                                gamma = y2 / max(0.01,y1)
                                X_train_spatial[c_spatial,:] = f
                                y_train_spatial[c_spatial] = gamma
                                c_spatial += 1
                        if(t > 0):
                            # Before
                            y1 = self.original_grid[i,j,t-1]
                            if(not(np.isnan(y1))):
                                f1 = self.feature_grid[i,j,:]
                                f3 = np.array([-1])
                                f = np.concatenate((f1,f2,f3))
                                f = f.reshape(1,len(f))
                                gamma = y2 / max(0.01,y1)
                                X_train_temporal[c_temporal,:] = f
                                y_train_temporal[c_temporal] = gamma
                                c_temporal += 1
                        if(t < d):
                            # After                       
                            y1 = self.original_grid[i,j,t+1]
                            if(not(np.isnan(y1))):
                                f1 = self.feature_grid[i,j,:]
                                f3 = np.array([1])
                                f = np.concatenate((f1,f2,f3))
                                f = f.reshape(1,len(f))
                                gamma = y2 / max(0.01,y1)
                                X_train_temporal[c_temporal,:] = f
                                y_train_temporal[c_temporal] = gamma
                                c_temporal += 1
                            
        self.model_spatial.fit(X_train_spatial,y_train_spatial)
        self.model_temporal.fit(X_train_temporal,y_train_temporal)