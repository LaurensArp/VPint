"""Module for MRP-based spatio-temporal interpolation.
"""

import numpy as np
import networkx as nx

from .MRP import SMRP, STMRP       
        
        
class WP_SMRP(SMRP):
    """
    Class for WP-SMRP, extending SMRP

    Attributes
    ----------
    original_grid : 2D numpy array
        the original grid supplied to be interpolated
    pred_grid : 2D numpy array
        interpolated version of original_grid
    feature_grid : 3D numpy array
        grid corresponding to original_grid, with feature vectors on the z-axis
    G : networkx directed graph
        graph representation of pred_grid
    model : sklearn-based prediction model
        user-supplied machine learning model used to predict weights

    Methods
    -------
    run():
        Runs WP-MRP
        
    train():
        Train supplied prediction model on subsampled data or a training set
        
    compute_confidence():
        compute an indication of uncertainty per pixel in pred_grid
    """    
    
    def __init__(self,grid,feature_grid,model,init_strategy='zero',max_gamma=np.inf,min_gamma=0):       
        super().__init__(grid,init_strategy=init_strategy)
        self.feature_grid = feature_grid.copy().astype(float)
        self.model = model 
        self.max_gamma = max_gamma
        self.min_gamma = min_gamma
    
            
    def run(self,iterations,method='predict'):
        """
        Runs WP-SMRP for the specified number of iterations. Creates a 3D (h,w,4) tensor val_grid, where the z-axis corresponds to a neighbour of each cell, and a 3D (h,w,4) weight tensor weight_grid, where the z-axis corresponds to the weights of every neighbour in val_grid's z-axis. The x and y axes of both tensors are stacked into 2D (h*w,4) matrices (one of which is transposed), after which the dot product is taken between both matrices, resulting in a (h*w,h*w) matrix. As we are only interested in multiplying the same row numbers with the same column numbers, we take the diagonal entries of the computed matrix to obtain a 1D (h*w) vector of updated values (we use numpy's einsum to do this efficiently, without wasting computation on extra dot products). This vector is then divided element-wise by a vector (flattened 2D grid) counting the number of neighbours of each cell, and we use the object's original_grid to replace wrongly updated known values to their original true values. We finally reshape this vector back to the original 2D pred_grid shape of (h,w).
        
        :param iterations: number of iterations used for the state value update function. If not specified, terminate once the maximal difference of a cell update dips below termination_threshold
        :param method: method for computing weights. Options: "predict" (using self.model), "cosine_similarity" (based on feature similarity), "exact" (compute average weight exactly for features)
        :returns: interpolated grid pred_grid
        """
               
        # Setup all this once
        
        height = self.pred_grid.shape[0]
        width = self.pred_grid.shape[1]
        
        h = height - 1
        w = width - 1
        
        neighbour_count_grid = np.zeros((height,width))
        weight_grid = np.zeros((height,width,4))
        val_grid = np.zeros((height,width,4))
        
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
        
        weight_matrix = weight_grid.reshape((height*width,4)).transpose()
        
        # Set neighbour count and weight grids
        
        for i in range(0,height):
            for j in range(0,width):
                if(i > 0 and i < height-1): 
                    if(j > 0 and j < width-1):
                        # Middle
                        neighbour_count_grid[i,j] = 4
                    elif(j == 0 or j == width-1):
                        # Left/right sides
                        neighbour_count_grid[i,j] = 3
                elif(i == 0 or i == height-1):
                    if(j > 0 and j < width-1):
                        # Top/bottom sides
                        neighbour_count_grid[i,j] = 3
                    elif(j == 0 or j == width-1):
                        # Corners
                        neighbour_count_grid[i,j] = 2
        
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
            gamma = self.model.predict(f)[0]
            gamma = max(self.min_gamma,min(gamma,self.max_gamma))
        elif(method == "cosine_similarity"):
            gamma = np.dot(f1,f2) / max(np.sum(f1) * np.sum(f2),0.01)
        elif(method == "exact"):
            f1_temp = f1.copy()
            f1_temp[f1_temp == 0] = 0.01
            gamma = np.mean(f2 / f1_temp) 
        else:
            print("Invalid method")
            intentionalcrash # TODO: start throwing proper exceptions...
        return(gamma)
        
    def run_old(self,iterations=None,termination_threshold=1e-4,method='predict'):
        """
        Runs WP-SMRP for the specified number of iterations.
        
        :param iterations: number of iterations used for the state value update function
        :param method: method for computing weights. Options: "predict" (using self.model), "cosine_similarity" (based on feature similarity), "exact" (compute average weight exactly for features)
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
                    for n1,n2,w in self.G.in_edges(n[0],data=True):
                        destination_node = self.G.nodes(data=True)[n1]
                        E_dest = destination_node['E']
                        r1 = self.G.nodes(data=True)[n1]['r']
                        c1 = self.G.nodes(data=True)[n1]['c']
                        r2 = self.G.nodes(data=True)[n2]['r']
                        c2 = self.G.nodes(data=True)[n2]['c']

                        f1 = self.feature_grid[r1,c1,:]
                        f2 = self.feature_grid[r2,c2,:]

                        if(method == "predict"):
                            f = np.concatenate((f1,f2))
                            f = f.reshape(1,len(f))
                            gamma = self.model.predict(f)[0]
                            gamma = max(self.min_gamma,min(gamma,self.max_gamma))
                        elif(method == "cosine_similarity"):
                            gamma = np.dot(f1,f2) / max(np.sum(f1) * np.sum(f2),0.01)
                        elif(method == "exact"):
                            f1_temp = f1.copy()
                            f1_temp[f1_temp == 0] = 0.01
                            gamma = np.mean(f2 / f1_temp) 
                        else:
                            print("Invalid method")
                            intentionalcrash # TODO: start throwing proper exceptions...

                        v_a = gamma * max(0.01,E_dest)
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
       
    
    def train(self,train_grid=None,train_features=None):
        """
        Trains WP-SMRP's weight prediction model on either subsampled
        data from original_grid and feature_grid, or a user-supplied 
        training grid with corresponding features.
        
        :param train_grid: optional user-specified training grid
        :param train_features: optional user-specified training feature grid
        """
    
        if(train_grid == None):
            train_grid = self.original_grid.copy()
        if(train_features == None):
            train_features = self.feature_grid.copy()
        
        # Compute true weight for all neighbour pairs with known values        
        true_gamma = {}
        num_viable = 0

        for n1,n2 in self.G.edges():
            y1 = self.G.nodes(data=True)[n1]['y']
            y2 = self.G.nodes(data=True)[n2]['y']
            if(not(np.isnan(y1) or np.isnan(y2))):
                y1 = self.G.nodes(data=True)[n1]['y']
                y2 = self.G.nodes(data=True)[n2]['y']
                true_weight = y2 / max(0.01,y1)
                true_gamma[(n1,n2)] = max(self.min_gamma,min(true_weight,self.max_gamma))
                num_viable += 1

        # Setup feature matrix and ground truth vector

        num_features = len(train_features[0][0]) * 2 

        y = np.zeros(num_viable)
        X = np.zeros((num_viable,num_features))

        # Iterate over edges

        i = 0
        for n1,n2,a in self.G.edges(data=True):
            y1 = self.G.nodes(data=True)[n1]['y']
            y2 = self.G.nodes(data=True)[n2]['y']
            if(not(np.isnan(y1) or np.isnan(y2))):
                gamma = true_gamma[(n1,n2)]
                r1 = self.G.nodes(data=True)[n1]['r']
                c1 = self.G.nodes(data=True)[n1]['c']
                r2 = self.G.nodes(data=True)[n2]['r']
                c2 = self.G.nodes(data=True)[n2]['c']
                
                f1 = train_features[r1,c1,:]
                f2 = train_features[r2,c2,:]
                f = np.concatenate((f1,f2))

                # Set features
                X[i,:] = f
                # Set label
                y[i] = true_gamma[(n1,n2)]

                i += 1

        # Train model

        self.model.fit(X,y)
        
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
        
        temp_MRP = WP_SMRP(new_grid,self.feature_grid,self.model)
        confidence_grid = temp_MRP.run(iterations)
        return(confidence_grid)
    
    

                
        
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
    G : networkx directed graph
        graph representation of pred_grid
    model : sklearn-based prediction model
        user-supplied machine learning model used to predict weights

    Methods
    -------
    run():
        Runs WP-MRP
        
    train():
        Train supplied prediction model on subsampled data or a training set
    """    
    def __init__(self,grid,feature_grid,model,auto_timesteps=False,max_gamma=np.inf,min_gamma=0):
        # Feature grid is a 3d grid, where x and y correspond to grid, and the z axis contains feature
        # vectors
        
        super(WP_STMRP, self).__init__(grid,auto_timesteps)
        self.feature_grid = feature_grid.copy().astype(float)
        self.model = model
        self.max_gamma = max_gamma
        self.min_gamma = min_gamma
        
        
    def run(self,iterations,method='predict'):
        """
        Runs WP-STMRP for the specified number of iterations. Creates a 4D (h,w,t,6) tensor val_grid, where the 4th axis corresponds to a neighbour of each cell, and a 4D (h,w,t,6) weight tensor weight_grid, where the 4th axis corresponds to the weights of every neighbour in val_grid's 4th axis. The x and y axes of both tensors are stacked into 2D (h*w*t,6) matrices (one of which is transposed), after which the dot product is taken between both matrices, resulting in a (h*w,h*w) matrix. As we are only interested in multiplying the same row numbers with the same column numbers, we take the diagonal entries of the computed matrix to obtain a 1D (h*w*t) vector of updated values (we use numpy's einsum to do this efficiently, without wasting computation on extra dot products). This vector is then divided element-wise by a vector (flattened 3D grid) counting the number of neighbours of each cell, and we use the object's original_grid to replace wrongly updated known values to their original true values. We finally reshape this vector back to the original 3D pred_grid shape of (h,w,t).
        
        :param iterations: number of iterations used for the state value update function. If not specified, terminate once the maximal difference of a cell update dips below termination_threshold
        :returns: interpolated grid pred_grid
        """

        # This one would also work for WP-MRP
        
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
                    vec = np.zeros(6)
                    f2 = self.feature_grid[i,j,:]
                    if(i > 0):
                        # Top
                        f1 = self.feature_grid[i-1,j,:]
                        f3 = np.array([0])
                        f = np.concatenate((f1,f2,f3))
                        vec[0] = self.predict_weight(f,method)
                    if(j < w):
                        # Right
                        f1 = self.feature_grid[i,j+1,:]
                        f3 = np.array([0])
                        f = np.concatenate((f1,f2,f3))
                        vec[1] = self.predict_weight(f,method)
                    if(i < h):
                        # Bottom
                        f1 = self.feature_grid[i+1,j,:]
                        f3 = np.array([0])
                        f = np.concatenate((f1,f2,f3))
                        vec[2] = self.predict_weight(f,method)
                    if(j > 0):
                        # Left
                        f1 = self.feature_grid[i,j-1,:]
                        f3 = np.array([0])
                        f = np.concatenate((f1,f2,f3))
                        vec[3] = self.predict_weight(f,method)
                    if(t > 0):
                        # Before
                        f1 = self.feature_grid[i,j,:]
                        f3 = np.array([-1])
                        f = np.concatenate((f1,f2,f3))
                        vec[4] = self.predict_weight(f,method)
                    if(t < d):
                        # After
                        f1 = self.feature_grid[i,j,:]
                        f3 = np.array([1])
                        f = np.concatenate((f1,f2,f3))
                        vec[5] = self.predict_weight(f,method)

                    weight_grid[i,j,t,:] = vec
        
        weight_matrix = weight_grid.reshape((height*width*depth,6)).transpose()
        
        # Set neighbour count and weight grids
        
        for i in range(0,height):
            for j in range(0,width):
                for t in range(0,depth):
                    if(i > 0 and i < height-1): 
                        if(j > 0 and j < width-1):
                            if(t > 0 and t < depth-1):
                                # Middle (s), middle (t)
                                neighbour_count_grid[i,j,t] = 6
                            else:
                                # Middle (s), edge (t)
                                neighbour_count_grid[i,j,t] = 5
                        elif(j == 0 or j == width-1):
                            if(t > 0 and t < depth-1):
                                # L/R edge (s), middle (t)
                                neighbour_count_grid[i,j,t] = 5
                            else:
                                # L/R edge (s), edge (t)
                                neighbour_count_grid[i,j,t] = 4
                    elif(i == 0 or i == height-1):
                        if(j > 0 and j < width-1):
                            if(t > 0 and t < depth-1):
                                # T/B edge (s), middle (t)
                                neighbour_count_grid[i,j,t] = 5
                            else:
                                # T/B edge (s), edge (t)
                                neighbour_count_grid[i,j,t] = 4

                        elif(j == 0 or j == width-1):
                            if(t > 0 and t < depth-1):
                                # Corner (s), middle (t)
                                neighbour_count_grid[i,j,t] = 4
                            else:
                                # Corner (s), edge (t)
                                neighbour_count_grid[i,j,t] = 3

        
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
            new_grid = new_grid.reshape((height,width,depth)) # Return to 2D grid

            self.pred_grid = new_grid
        return(self.pred_grid)
    
    
    def predict_weight(self,f,method):
        """
        Predict the weight between two cells based on feature vectors f1 and f2, using either self.model, cosine similarity or the average exact weight computed from the two feature vectors.
        
        :param f1: feature vector for the neighbouring cell
        :param f2: feature vector for the cell of interest
        :param method: method for computing weights. Options: "predict" (using self.model), "cosine_similarity" (based on feature similarity), "exact" (compute average weight exactly for features)
        :returns: predicted weight gamma
        """
        if(method == "predict"):
            f = f.reshape(1,len(f))
            gamma = self.model.predict(f)[0]
            gamma = max(self.min_gamma,min(gamma,self.max_gamma))
        elif(method == "cosine_similarity"):
            gamma = np.dot(f1,f2) / max(np.sum(f1) * np.sum(f2),0.01)
        elif(method == "exact"):
            f1_temp = f1.copy()
            f1_temp[f1_temp == 0] = 0.01
            gamma = np.mean(f2 / f1_temp) 
        else:
            print("Invalid method")
            intentionalcrash # TODO: start throwing proper exceptions...
        return(gamma)
    
            
    def run_old(self,iterations):
        """
        Runs WP-STMRP for the specified number of iterations.
        
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
                    for n1,n2,w in self.G.in_edges(n[0],data=True):
                        destination_node = self.G.nodes(data=True)[n1]
                        E_dest = destination_node['E']
                        r1 = self.G.nodes(data=True)[n1]['r']
                        c1 = self.G.nodes(data=True)[n1]['c']
                        r2 = self.G.nodes(data=True)[n2]['r']
                        c2 = self.G.nodes(data=True)[n2]['c']

                        f1 = self.feature_grid[r1,c1,:]
                        f2 = self.feature_grid[r2,c2,:]

                        if(destination_node['t'] != n[1]['y']):
                            f3 = np.array([1])
                        else:
                            f3 = np.array([0])
                        f = np.concatenate((f1,f2,f3))
                        f = f.reshape(1,len(f))

                        v_a = self.model.predict(f)[0] * E_dest
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
    
    
    
    def train(self,train_grid=None,train_features=None):
        """
        Trains WP-STMRP's weight prediction model on either subsampled
        data from original_grid and feature_grid, or a user-supplied 
        training grid with corresponding features.
        
        :param train_grid: optional user-specified training grid
        :param train_features: optional user-specified training feature grid
        """
        if(train_grid == None):
            train_grid = self.original_grid.copy()
        if(train_features == None):
            train_features = self.feature_grid.copy()
        
        # Compute true weight for all neighbour pairs with known values        
        true_gamma = {}
        num_viable = 0

        for n1,n2 in self.G.edges():
            y1 = self.G.nodes(data=True)[n1]['y']
            y2 = self.G.nodes(data=True)[n2]['y']
            if(not(np.isnan(y1) or np.isnan(y2))):
                y1 = self.G.nodes(data=True)[n1]['y']
                y2 = self.G.nodes(data=True)[n2]['y']
                true_weight = y2 / max(0.01,y1)
                #true_gamma[(n1,n2)] = true_weight
                true_gamma[(n1,n2)] = max(self.min_gamma,min(true_weight,self.max_gamma))
                num_viable += 1

        # Setup feature matrix and ground truth vector

        num_features = len(train_features[0][0]) * 2 + 1

        y = np.zeros(num_viable)
        X = np.zeros((num_viable,num_features))

        # Iterate over edges

        i = 0
        for n1,n2,a in self.G.edges(data=True):
            y1 = self.G.nodes(data=True)[n1]['y']
            y2 = self.G.nodes(data=True)[n2]['y']
            if(not(np.isnan(y1) or np.isnan(y2))):
                gamma = true_gamma[(n1,n2)]
                r1 = self.G.nodes(data=True)[n1]['r']
                c1 = self.G.nodes(data=True)[n1]['c']
                r2 = self.G.nodes(data=True)[n2]['r']
                c2 = self.G.nodes(data=True)[n2]['c']
                
                f1 = train_features[r1,c1,:]
                f2 = train_features[r2,c2,:]
                if(self.G.nodes(data=True)[n1]['t'] > self.G.nodes(data=True)[n2]['t']):
                    f3 = np.array([1])
                elif(self.G.nodes(data=True)[n1]['t'] < self.G.nodes(data=True)[n2]['t']):
                    f3 = np.array([1])
                else:
                    f3 = np.array([0])
                
                f = np.concatenate((f1,f2,f3))

                # Set features
                X[i,:] = f
                # Set label
                y[i] = true_gamma[(n1,n2)]

                i += 1

        # Train model

        self.model.fit(X,y)