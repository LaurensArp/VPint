

import networkx as nx
import numpy as np
import pandas as pd

from .SAR import *
from .MA import *

from sklearn.metrics import mean_absolute_error as mae
from sklearn.linear_model import LinearRegression

import autosklearn.regression





def train_ARMA(G_train,W_train,y_mean_train,autoskl_params1,autoskl_params2):
    # Essentially identical to MA training code, but using SAR instead of basic
    # regression for its errors, and adds SAR feature to final dataset. 

    # 1) Train SAR model on known labels
    
    

    X_train, y_train, is_non_hidden_train = basic_feature_matrix(G_train)
    lag_train = lagged_feature(G_train,W_train,y_mean_train)
    X_train = add_lagged_feature(X_train,lag_train)
    X_train_SAR, y_train_SAR = remove_hidden(X_train,y_train,is_non_hidden_train)
    
    # Train

    feature_types = (['numerical'] * len(X_train_SAR[0,:]))

    m_SAR = autosklearn.regression.AutoSklearnRegressor(
        time_left_for_this_task=autoskl_params1['max_time'],
        per_run_time_limit=autoskl_params1['max_time_per_run'],
        tmp_folder=autoskl_params1['tmp_folder'],
        output_folder=autoskl_params1['output_folder'],
        delete_tmp_folder_after_terminate=autoskl_params1['delete_temp'],
        delete_output_folder_after_terminate=autoskl_params1['delete_output'],
    )

    m_SAR.fit(X_train_SAR, y_train_SAR, feat_type=feature_types)

    

    # 2) Run SAR on known labels
    
    pred_SAR = m_SAR.predict(X_train)
    mean_error = np.mean(pred_SAR)
    
    # 3) Compute mean SAR errors
    
    e_train = np.zeros(len(G_train.nodes))
    
    i = 0
    for n in G_train.nodes(data=True):
        if(n[1]['hidden']):
            e_train[i] = mean_error
        else:
            s = np.dot(W_train[i,:],y_train)
            A = n[1]['A']
            vec = np.append(A,[s])
            vec = vec.reshape(1,len(vec))
            pred = m_SAR.predict(vec)[0]
            e_train[i] = pred - n[1]['label']    
        i += 1

    
    # 4) Compute spatial error term feature
    
    f = np.dot(W_train,e_train)
    
    # 5) Add to SAR features
    
    X_train_df = pd.DataFrame(X_train)
    X_train_df['e'] = f
    X_train = X_train_df.values
    
    
    # 6) Run autosklearn on ARMA        
        
    feature_types = (['numerical'] * len(X_train[0,:]))

    automl = autosklearn.regression.AutoSklearnRegressor(
        time_left_for_this_task=autoskl_params2['max_time'],
        per_run_time_limit=autoskl_params2['max_time_per_run'],
        tmp_folder=autoskl_params2['tmp_folder'],
        output_folder=autoskl_params2['output_folder'],
        delete_tmp_folder_after_terminate=autoskl_params2['delete_temp'],
        delete_output_folder_after_terminate=autoskl_params2['delete_output'],
    )

    automl.fit(X_train, y_train, feat_type=feature_types)
    return(m_SAR,automl)
    
    
    
    

def test_ARMA(G_test,W_test,y_mean_test,m_SAR,automl):
    # Same as MA test code, except now using SAR instead of basic regression


    # Run on test set
    

    X_test, y_test, is_non_hidden_test = basic_feature_matrix(G_test)
    lag_test = lagged_feature(G_test,W_test,y_mean_test)
    X_test = add_lagged_feature(X_test,lag_test)

    e_test = np.zeros(len(G_test.nodes))
               
    pred_SAR = m_SAR.predict(X_test)
    mean_error = np.mean(pred_SAR - y_test)
    
    i = 0
    for n in G_test.nodes(data=True):
        if(n[1]['hidden']):
            e_test[i] = mean_error
        else:
            s = np.dot(W_test[i,:],y_test)
            A = n[1]['A']
            vec = np.append(A,[s])
            vec = vec.reshape(1,len(vec))
            pred = m_SAR.predict(vec)[0]
            e_test[i] = pred - n[1]['label']    
        i += 1

    f = np.dot(W_test,e_test)
    X_test_df = pd.DataFrame(X_test)
    X_test_df['e'] = f
    X_test = X_test_df.values
                      
    pred = automl.predict(X_test)
    error = mae(pred,y_test)
                      
    return(error)


    
    
    
    
