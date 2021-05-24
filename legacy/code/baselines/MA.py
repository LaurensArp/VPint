

import networkx as nx
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error as mae
from sklearn.linear_model import LinearRegression

import autosklearn.regression





def train_MA(G_train,W_train,autoskl_params):
    # Run MA baseline. Using "MA by AR" approach, where errors are obtained
    # by running a model with no MA term first. We used canonical linear 
    # regression for this temporary model; the actual MA model uses auto-sklearn.


    # 1) Train LR model on known labels

    num_features = -1
    num_instances = 0
    for n in G_train.nodes(data=True):
        num_features = len(n[1]['A'])
        break
        
        
    X_train = np.zeros((len(G_train.nodes),num_features))
    y_train = np.zeros(len(G_train.nodes))
        
    i = 0
    for n in G_train.nodes(data=True):
        if(not(n[1]['hidden'])):
            num_instances += 1
            
        X_train[i,:] = n[1]['A']
        y_train[i] = n[1]['label']
        i += 1
        
    X_train_LR = np.zeros((num_instances,num_features))   
    y_train_LR = np.zeros(num_instances)
    e_train = np.zeros(len(G_train.nodes))
    
    i = 0
    for n in G_train.nodes(data=True):
        if(not(n[1]['hidden'])):
            X_train_LR[i,:] = n[1]['A']
            y_train_LR[i] = n[1]['label']
            i += 1
            
    
    m_LR = LinearRegression()
    m_LR.fit(X_train_LR,y_train_LR)

    # 2) Run LR on known labels
    
    pred_LR = m_LR.predict(X_train_LR)
    mean_error = np.mean(pred_LR)
    
    # 3) Compute mean LR errors
    
    e_train = np.zeros(len(G_train.nodes))
    
    i = 0
    for n in G_train.nodes(data=True):
        if(n[1]['hidden']):
            e_train[i] = mean_error
        else:
            pred = m_LR.predict([n[1]['A']])[0]
            e_train[i] = pred - n[1]['label']    
        i += 1

    
    # 4) Compute spatial error term feature
    
    f = np.dot(W_train,e_train)
    
    # 5) Add to LR features
    
    X_train_df = pd.DataFrame(X_train)
    X_train_df['e'] = f
    X_train = X_train_df.values
    
    # 6) Run autosklearn        
        
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
    return(m_LR,automl)
    

def test_MA(G_test,W_test,m_LR,automl):
    # Test a pre-trained MA model. Requires both the temporary model to
    # compute errors, and the final MA model.

    # Run on test set
    
    num_features = -1
    for n in G_test.nodes(data=True):
        num_features = len(n[1]['A'])
        break
    
    # Create tabular dataset
    X_test = np.zeros((len(G_test.nodes),num_features))
    y_test = np.zeros(len(G_test.nodes))
    
    e_test = np.zeros(len(G_test.nodes))
                      
                      
    i = 0
    for n in G_test.nodes(data=True):
        X_test[i,:] = n[1]['A']
        y_test[i] = n[1]['label']
        i += 1
               
    pred_LR = m_LR.predict(X_test)
    mean_error = np.mean(pred_LR - y_test)
    
    i = 0
    for n in G_test.nodes(data=True):
        if(n[1]['hidden']):
            e_test[i] = mean_error
        else:
            pred = m_LR.predict([n[1]['A']])[0]
            e_test[i] = pred - n[1]['label']    
        i += 1

    # Run MA
    f = np.dot(W_test,e_test)
    X_test_df = pd.DataFrame(X_test)
    X_test_df['e'] = f
    X_test = X_test_df.values
                      
    pred = automl.predict(X_test)
    error = mae(pred,y_test)
                      
    return(error)



    
    
    
    
