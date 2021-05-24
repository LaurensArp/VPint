import numpy as np
import networkx as nx
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

from sklearn.svm import SVR
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error

import autosklearn.regression

import pykrige.kriging_tools as kt
from pykrige.ok import OrdinaryKriging
from pykrige.uk import UniversalKriging

from mrp.mrp_misc import initialise_MRP

def create_simple_matrix(G):
    # Transform graph into tabular dataset where features are node attribute
    # vectors, and true values are the matched y* values

    h = len(G.nodes)
    w = -1
    for n in G.nodes(data=True):
        w = len(n[1]['A'])
        break
        
    X = np.zeros((h,w))
    y = np.zeros(h)
    
    i = 0
    for n in G.nodes(data=True):
        X[i,:] = n[1]['A']
        y[i] = n[1]['label']
        i += 1
    
    return(X,y)
    
def create_basic_grid(G,height,width,labels="all"):
    # Turns graph into a grid. Used initially for exploratory visualisation, but
    # used by some baselines to compute errors. Could be removed to use a more direct
    # way of computing errors, but since this was used to obtain our results, we are
    # providing the code just in case.

    # Possible values labels: "all", "unhidden", "labels_and_predictions", "predictions"

    grid = np.zeros((height,width))
    
    if(labels=="all"):
        for n in G.nodes(data=True):
            x = n[1]['x']
            y = n[1]['y']
            label = n[1]['label']
            grid[y,x] = label

    elif(labels=="unhidden"):
        for n in G.nodes(data=True):
            x = n[1]['x']
            y = n[1]['y']
            if(not(n[1]['hidden'])):
                label = n[1]['label']
                grid[y,x] = label
            else:
                grid[y,x] = np.nan
                
                
    elif(labels=="hidden"):
        for n in G.nodes(data=True):
            x = n[1]['x']
            y = n[1]['y']
            if(n[1]['hidden']):
                label = n[1]['label']
                grid[y,x] = label
            else:
                grid[y,x] = np.nan
                
    elif(labels=="labels_and_predictions"):
        for n in G.nodes(data=True):
            x = n[1]['x']
            y = n[1]['y']
            if(not(n[1]['hidden'])):
                label = n[1]['label']
                grid[y,x] = label
            else:
                E = n[1]['E']
                grid[y,x] = E
                
    elif(labels=="predictions"):
        for n in G.nodes(data=True):
            x = n[1]['x']
            y = n[1]['y']
            if(not(n[1]['hidden'])):
                grid[y,x] = np.nan
            else:
                E = n[1]['E']
                grid[y,x] = E
                
    else:
        print("Invalid label set.")
        
    grid = np.flip(grid,axis=0) # Indexing is reversed
    return(grid)


def simple_to_MRP(G,model,neighbours):
    # Used by some baselines to use MRP-based evaluation methods with
    # a pre-defined (non-MRP) prediction model. 

    MRP = G.copy()
    MRP = initialise_MRP(MRP)
    for n in MRP.nodes(data=True):
        A = n[1]['A']
        E = np.nan
        if(neighbours):
            j = 0
            in_edges = G.in_edges(n[0],data=True)
            
            vec = np.zeros(4)
            for n1,n2,w in G.in_edges(n[0],data=True):
                origin_node = G.nodes(data=True)[n1]
                hidden = origin_node['hidden']
                if(not(hidden)):
                    vec[j] = origin_node['label']
                    j += 1
                    
            if(j > 0):
                while(j <= 3):
                    so_far = vec[0:j]
                    vec[j] = np.mean(so_far)
                    j += 1
            
            combined = np.concatenate((A,vec))
            E = model.predict(combined.reshape(1,len(combined)))[0]
            
            
        else:
            E = model.predict(A.reshape(1,len(A)))[0]
        nx.set_node_attributes(MRP, {n[0]:E}, 'E')
    return(MRP)



def simple_regression_auto(G_train,G_test,autoskl_params,verbose=False):
    # Run the basic regression baseline, using auto-sklearn

    X_train,y_train = create_simple_matrix(G_train)
    X_test,y_test = create_simple_matrix(G_test)
    
    feature_types = (['numerical'] * len(X_train[0,:]))
    
    automl = autosklearn.regression.AutoSklearnRegressor(
        time_left_for_this_task=autoskl_params['max_time'],
        per_run_time_limit=autoskl_params['max_time_per_run'],
        tmp_folder=autoskl_params['tmp_folder'],
        output_folder=autoskl_params['output_folder'],
        delete_tmp_folder_after_terminate=autoskl_params['delete_temp'],
        delete_output_folder_after_terminate=autoskl_params['delete_output'],
    )
    
    automl.fit(X_train, y_train, feat_type=feature_types)
    
    pred = automl.predict(X_test)
    
    if(verbose):
        print("MAE: ",mean_absolute_error(y_test,pred))
        print("Mean predictions: ",np.mean(pred))
        print("Standard deviation predictions: ",np.std(pred))
    
    return(pred,y_test,automl)
    
    
def simple_regression_auto_self(G_train,autoskl_params,verbose=False):
    # For same-city supervision, this is a separate function to train
    # basic regression using sub-sampling of known true values
    
    X_train,y_train = create_simple_matrix(G_train)
    
    feature_types = (['numerical'] * len(X_train[0,:]))
    
    automl = autosklearn.regression.AutoSklearnRegressor(
        time_left_for_this_task=autoskl_params['max_time'],
        per_run_time_limit=autoskl_params['max_time_per_run'],
        tmp_folder=autoskl_params['tmp_folder'],
        output_folder=autoskl_params['output_folder'],
        delete_tmp_folder_after_terminate=autoskl_params['delete_temp'],
        delete_output_folder_after_terminate=autoskl_params['delete_output'],
    )
    
    automl.fit(X_train, y_train, feat_type=feature_types)
    
  
    return(automl)

    
    
def baseline_ordinary_kriging(G,height,width,variogram_model='linear',verbose=False):
    # Ordinary kriging baseline, code based on examples at 
    # https://geostat-framework.readthedocs.io/projects/pykrige/en/stable/

    # Initialise grid
    gridx = np.arange(0.0, float(width), 1.0)
    gridy = np.arange(0.0, float(height), 1.0)
    
    # Initialise value table, add data from graph
    vals = []
    for n in G.nodes(data=True):
        if(not(n[1]['hidden'])):
            x = n[1]['x']
            y = n[1]['y']
            label = n[1]['label']
            vals.append([x,y,label])
            
    data = np.array(vals)
    OK = OrdinaryKriging(data[:, 0], data[:, 1], data[:, 2], variogram_model=variogram_model,
                            verbose=False, enable_plotting=False)
                            
    kriged_grid, var_grid = OK.execute('grid', gridx, gridy)
    
    
    
    
    # Get label grids; this is legacy and could be done much more simply
    hidden_label_grid = create_basic_grid(G,height,width,labels="hidden")
    label_grid = create_basic_grid(G,height,width,labels="all")


    # Compute error; needs iteration instead of flatten to ignore known labels
    
    # We did not end up using percentage errors; again, this code is fairly old
    pred = []
    y = []
    percentage_grid = np.zeros((height,width))
    for i in range(0,len(kriged_grid)):
        for j in range(0,len(kriged_grid[0,:])):
            pred.append(kriged_grid[i,j])
            y.append(label_grid[i,j])
            
            p_err = np.nan
            if(label_grid[i,j] > 0):
                p_err = abs(kriged_grid[i,j] - label_grid[i,j]) / label_grid[i,j] * 100
            percentage_grid[i,j] = p_err
            
    percentage_errors = percentage_grid.flatten()
            
    pred = np.array(pred)
    y = np.array(y)

    loss = mean_absolute_error(y,pred)
    
    if(verbose):
        # Visualise

        print("Training set")

        plt.imshow(label_grid)
        plt.title('All labels')
        plt.colorbar()
        plt.xlabel("X index")
        plt.ylabel("Y index")
        plt.show()

        plt.imshow(hidden_label_grid)
        plt.title('Hidden labels')
        plt.colorbar()
        plt.xlabel("X index")
        plt.ylabel("Y index")
        plt.show()

        plt.imshow(kriged_grid)
        plt.title('Predictions')
        plt.colorbar()
        plt.xlabel("X index")
        plt.ylabel("Y index")
        plt.show()
        
        plt.imshow(percentage_grid)
        plt.title('Percentage error')
        plt.colorbar()
        plt.xlabel("X index")
        plt.ylabel("Y index")
        plt.show()
    
    
    return(loss,kriged_grid,var_grid,label_grid,percentage_grid,pred,y,percentage_errors)

    
    
def baseline_universal_kriging(G,height,width,variogram_model='linear',drift_terms=['regional_linear'],verbose=False):
    # Comments for ordinary kriging apply here as well
    
    # Based on examples at 
    # https://geostat-framework.readthedocs.io/projects/pykrige/en/stable/

    # Initialise grid
    gridx = np.arange(0.0, float(width), 1.0)
    gridy = np.arange(0.0, float(height), 1.0)
    
    # Initialise value table, add data from graph
    vals = []
    for n in G.nodes(data=True):
        if(not(n[1]['hidden'])):
            x = n[1]['x']
            y = n[1]['y']
            label = n[1]['label']
            vals.append([x,y,label])
            
    data = np.array(vals)
    UK = UniversalKriging(data[:, 0], data[:, 1], data[:, 2], variogram_model=variogram_model,
                            drift_terms=drift_terms)
                            
    kriged_grid, var_grid = UK.execute('grid', gridx, gridy)
    
    # Get label grids
    hidden_label_grid = create_basic_grid(G,height,width,labels="hidden")
    label_grid = create_basic_grid(G,height,width,labels="all")


    # Compute error; needs iteration instead of flatten to ignore known labels
    pred = []
    y = []
    percentage_grid = np.zeros((height,width))
    for i in range(0,len(kriged_grid)):
        for j in range(0,len(kriged_grid[0,:])):
            pred.append(kriged_grid[i,j])
            y.append(label_grid[i,j])
            
            p_err = np.nan
            if(label_grid[i,j] > 0):
                p_err = (kriged_grid[i,j] - label_grid[i,j]) / label_grid[i,j]
            percentage_grid[i,j] = p_err
            
    percentage_errors = percentage_grid.flatten()
    
    pred = np.array(pred)
    y = np.array(y)

    loss = mean_absolute_error(y,pred)
    
    if(verbose):
        # Visualise

        print("Training set")

        plt.imshow(label_grid)
        plt.title('All labels')
        plt.colorbar()
        plt.xlabel("X index")
        plt.ylabel("Y index")
        plt.show()

        plt.imshow(hidden_label_grid)
        plt.title('Hidden labels')
        plt.colorbar()
        plt.xlabel("X index")
        plt.ylabel("Y index")
        plt.show()

        plt.imshow(kriged_grid)
        plt.title('Predictions')
        plt.colorbar()
        plt.xlabel("X index")
        plt.ylabel("Y index")
        plt.show()
        
        plt.imshow(percentage_grid)
        plt.title('Percentage error')
        plt.colorbar()
        plt.xlabel("X index")
        plt.ylabel("Y index")
        plt.show()
    
    
    return(loss,kriged_grid,var_grid,label_grid,percentage_grid,pred,y,percentage_errors)
    