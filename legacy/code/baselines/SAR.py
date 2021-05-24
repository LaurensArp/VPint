

import networkx as nx
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error as mae
from sklearn import linear_model

import autosklearn.regression



def weight_matrix_rook(G):
    # Create a rook-based weight matrix for a given graph. Called W here, referred
    # to as M in the paper to avoid confusion with other variables. We also compute
    # mean labels to facilitate mean imputation later on, to avoid unnecessary extra
    # loops.

    W = np.zeros((len(G.nodes),len(G.nodes)))

    # Convenient for matrix indexing
    indices = {}
    i = 0
    for n in G.nodes(data=True):
        indices[n[0]] = i
        i += 1


    # Create matrix (rook binary weighting). Finding mean label not technically part of it, but
    # it's computationally convenient
    y_tot = 0
    y_count = 0

    for n in G.nodes(data=True):
        # Weight matrix
        r = indices[n[0]]
        neighbour_iter = G.neighbors(n[0])
        for neigh in neighbour_iter:
            c = indices[neigh]
            W[r,c] = 1

        # Mean y
        if(not(n[1]['hidden'])):
            y_tot += n[1]['label']
            y_count += 1

    y_mean = y_tot / y_count

    return(W,y_mean)
    
    
    
def basic_feature_matrix(G):
    # Create a basic feature matrix as would be used by basic regression


    # Adding in y and number of non-hidden to avoid unnecessary computation
    
    num_features = -1
    for n in G.nodes(data=True):
        num_features = len(n[1]['A'])
        break
        
    X = np.zeros((len(G.nodes),num_features))
    y = np.zeros(len(G.nodes))
    is_non_hidden = np.zeros(len(G.nodes))
    
    r = 0
    for n in G.nodes(data=True):
        A = n[1]['A']
        X[r,:] = A
        y[r] = n[1]['label']
        if(n[1]['hidden']):
            is_non_hidden[r] = 0
        else:
            is_non_hidden[r] = 1
        r += 1
    
    return(X,y,is_non_hidden)
    
    
    
def lagged_feature(G,W,y_mean):
    # This feature sets SAR apart from regular regression. It is essentially
    # the weighted average of neighbouring true values based on the weight 
    # matrix.

    label_vec = np.zeros((len(G),1))
    
    # Create label vector for W * Y multiplication
    i = 0
    for n in G.nodes(data=True):
        if(n[1]['hidden']):
            label_vec[i] = y_mean
        else:
            label_vec[i] = n[1]['label']
            
    # Multiply
    lagged_feature_vec = np.dot(W,label_vec)
    
    return(lagged_feature_vec)
    
    
def add_lagged_feature(X,lagged):
    # Extend feature matrix with new variable
    df = pd.DataFrame(X)
    df['lag'] = lagged

    X_new = df.values
    
    return(X_new)
    
def remove_hidden(X,y,is_non_hidden):
    # Don't train on hidden (unknown) nodes

    df = pd.DataFrame(X).iloc[is_non_hidden>0]
    X_new = df.values
    y_new = pd.DataFrame(y.reshape(len(y),1)).iloc[is_non_hidden>0].values # ugly but does the job
    
    return(X_new,y_new)
    
   
  
def spatial_lag_auto(X_train,y_train,X_test,y_test,autoskl_params):
    # Basically identical to basic regression code; lagged feature has already
    # been added to the feature matrices


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

    error = mae(pred,y_test)
    
    return(error)




    
    
    
    
