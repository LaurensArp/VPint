"""Module for MRP-based spatio-temporal interpolation.
"""

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
            
    def run(self,iterations):
        """
        Runs SD-STMRP for the specified number of iterations.
        
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
        
        
        
class WP_STMRP(STMRP):
    """
    Class for WP-STMRP, extending STMRP

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
    """    
    def __init__(self,grid,feature_grid,model,auto_timestamps=False):
        # Feature grid is a 3d grid, where x and y correspond to grid, and the z axis contains feature
        # vectors
        
        super(WP_STMRP, self).__init__(grid,auto_timestamps)
        self.feature_grid = feature_grid.copy()
        self.model = model  
    
            
    def run(self,iterations):
        """
        Runs WP-STMRP for the specified number of iterations.
        
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
                    
                    if(destination_node['t'] != n[1]['y']):
                        f3 = np.array([1])
                    else:
                        f3 = np.array([0])
                    f = np.concatenate((f1,f2,f3))
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