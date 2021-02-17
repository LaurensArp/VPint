"""Module for MRP-based spatial interpolation.
"""

import numpy as np
import networkx as nx

class SMRP:
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
        
        
    def mean_absolute_error(self,true_grid,gridded=False):
        """
        Compute the mean absolute error of pred_grid given true_grid as ground truth
        
        :param true_grid: ground truth for all grid cells
        :param gridded: optional Boolean specifying whether to return an error grid with the MAE
        :returns: mean absolute error, optionally a tuple of MAE and per-cell error
        """
        height = self.pred_grid.shape[0]
        width = self.pred_grid.shape[1]       
        error_grid = np.zeros((height,width))
        
        e = 0
        c = 0
        for i in range(0,height):
            for j in range(0,width):
                if(np.isnan(self.original_grid[i][j])):
                    err = abs(self.pred_grid[i][j] - true_grid[i][j])
                    error_grid[i][j] = err
                    e += err
                    c += 1
                else:
                    error_grid[i][j] = np.nan
        
        mae = e/c
        if(gridded):
            result = (mae,error_grid)
        else:
            result = mae
            
        return(result)
        
        
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
    
    def __init__(self,grid,gamma=0.9):
        super().__init__(grid)
        self.gamma = gamma
    
    
    def set_gamma(self,gamma):
        """
        Sets gamma to the manually supplied value.
        
        :param gamma: user-supplied gamma value
        """
        self.gamma = gamma
    
            
    def run(self,iterations):
        """
        Runs SD-SMRP for the specified number of iterations.
        
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
                                                  
            best_loss = np.inf
            best_gamma = 0.9
            
            for ep in range(0,search_epochs):
                # Random search for best gamma for search_epochs iterations
                
                temp_MRP = SD_SMRP(sub_grid)
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
    
    def __init__(self,grid,feature_grid,model):       
        super().__init__(grid)
        self.feature_grid = feature_grid.copy()
        self.model = model 
    
            
    def run(self,iterations):
        """
        Runs WP-SMRP for the specified number of iterations.
        
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
                    r1 = self.G.nodes(data=True)[n1]['r']
                    c1 = self.G.nodes(data=True)[n1]['c']
                    r2 = self.G.nodes(data=True)[n2]['r']
                    c2 = self.G.nodes(data=True)[n2]['c']

                    f1 = self.feature_grid[r1,c1,:]
                    f2 = self.feature_grid[r2,c2,:]
                    f = np.concatenate((f1,f2))
                    f = f.reshape(1,len(f))
                    
                    v_a = self.model.predict(f)[0] * E_dest
                    v_a_sum += v_a
                E_new = v_a_sum / len(self.G.in_edges(n[0]))
                nx.set_node_attributes(self.G,{n[0]:E_new},'E')

            else:
                nx.set_node_attributes(self.G,{n[0]:y},'E')
        
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
                true_gamma[(n1,n2)] = true_weight
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