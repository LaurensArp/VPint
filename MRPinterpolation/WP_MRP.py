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
    
            
    def run(self,iterations=None,termination_threshold=1e-4,method='predict'):
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
    def __init__(self,grid,feature_grid,model,auto_timestamps=False):
        # Feature grid is a 3d grid, where x and y correspond to grid, and the z axis contains feature
        # vectors
        
        super(WP_STMRP, self).__init__(grid,auto_timestamps)
        self.feature_grid = feature_grid.copy().astype(float)
        self.model = model  
    
            
    def run(self,iterations):
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
                true_gamma[(n1,n2)] = true_weight
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
                if(self.G.nodes(data=True)[n1]['t'] != self.G.nodes(data=True)[n2]['t']):
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