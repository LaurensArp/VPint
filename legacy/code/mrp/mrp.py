

import numpy as np
import networkx as nx
import math

from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error

import autosklearn.regression

import cma

from .mrp_misc import run_MRP, initialise_MRP, initialise_W




############# Useful shared functions ####################################


def get_validation_vectors(G):
    # Return vector of predictions y hat and vector of true values y*
    
    pred = np.zeros(len(G.nodes))
    labels = np.zeros(len(G.nodes))

    i = 0
    for n in G.nodes(data=True):
        if(n[1]['hidden']):
            labels[i] = n[1]['label']
            pred[i] = n[1]['E']
            
        else:
            # Set to nan to remove later (don't count known measurements for error)
            pred[i] = np.nan
            labels[i] = np.nan
            
        i += 1
        
    # Remove nans
    
    pred = pred[np.logical_not(np.isnan(pred))]
    labels = labels[np.logical_not(np.isnan(labels))]
        
    return(pred,labels)
    
def simple_train_test_split(X,labels,train_proportion):
    # Split a feature matrix and ground truth vector into train- and
    # test sets at a specified proportion

    indices = np.arange(len(X))
    np.random.shuffle(indices)

    X = X[indices]
    labels = labels[indices]

    split_index = int(train_proportion*len(X))

    X_train = X[0:split_index,:]
    X_test = X[split_index+1:len(X)-1,:]

    y_train = labels[0:split_index]
    y_test = labels[split_index+1:len(labels)-1]
    
    return(X_train,X_test,y_train,y_test)
    
    
    
def normalise_attributes(G,method):
    # Attribute normalisation, based on https://en.wikipedia.org/wiki/Feature_scaling

    super_H = G.copy()
    
    # Unit length, fastest, probably worst
    if(method == "unit"):
        for n in super_H.nodes(data=True):
            A = n[1]['A']
            length = sum(A)
            if(length > 0):
                A = A / sum(A)
                nx.set_node_attributes(super_H,{n[0]:A},'A')
    
    # Z-score normalisation, common in ML, needs 2 passes over all nodes
    elif(method == "z_score"):
        # Get length of A
        num_attrs = 0
        for n in super_H.nodes(data=True):
            num_attrs = len(n[1]['A'])
            break
        
        # Create matrix of |V| rows, |A| cols
        X = np.zeros((len(super_H.nodes),num_attrs))
        i = 0
        for n in super_H.nodes(data=True):
            for j in range(0,len(n[1]['A'])):
                X[i,j] = n[1]['A'][j]
            i += 1
            
        means = np.zeros(num_attrs)
        stds = np.zeros(num_attrs)
        
        for i in range(0,num_attrs):
            means[i] = np.mean(X[:,i])
            stds[i] = np.std(X[:,i])
        
        for n in super_H.nodes(data=True):
            A = n[1]['A']
            A_new = np.zeros(num_attrs)
            for i in range(0,num_attrs):
                A_new[i] = (A[i] - means[i]) / stds[i]
            nx.set_node_attributes(super_H,{n[0]:A_new},'A')
        
    # Mean normalisation, needs 2 passes
    elif(method == "mean_norm"):
        # Get length of A
        num_attrs = 0
        for n in super_H.nodes(data=True):
            num_attrs = len(n[1]['A'])
            break
        
        # Create matrix of |V| rows, |A| cols
        X = np.zeros((len(super_H.nodes),num_attrs))
        i = 0
        for n in super_H.nodes(data=True):
            for j in range(0,len(n[1]['A'])):
                X[i,j] = n[1]['A'][j]
            i += 1
            
        means = np.zeros(num_attrs)
        maxs = np.zeros(num_attrs)
        mins = np.zeros(num_attrs)
        
        for i in range(0,num_attrs):
            means[i] = np.mean(X[:,i])
            maxs[i] = np.max(X[:,i])
            mins[i] = np.min(X[:,i])
        
        for n in super_H.nodes(data=True):
            A = n[1]['A']
            A_new = np.zeros(num_attrs)
            for i in range(0,num_attrs):
                A_new[i] = (A[i] - means[i]) / (maxs[i] - mins[i])
            nx.set_node_attributes(super_H,{n[0]:A_new},'A')
            
    elif(method == "none"):
            pass
    else:
        print("Invalid normalisation method.")
        
    return(super_H)
    


    
############# Controller functions ####################################
    

    
def normal_MRP(G,MRP_iter,MRP_gamma,verbose=False,debug=False):
    # Run SD-MRP. What we call W in the code is gamma in the paper
   
    MRP = initialise_MRP(G)
    W = initialise_W(MRP,default_weight=MRP_gamma)
    

    # Run MRP
    MRP = run_MRP(MRP,W,MRP_iter,debug=False)

    # Compute error
    pred,labels = get_validation_vectors(MRP)
    total_error = mean_absolute_error(labels,pred)   
    if(verbose):
        print("Loss: " + str(total_error))


    if(debug):
        print(total_error)
        
    return(MRP,total_error,pred)
    
    
def normal_MRP_self_supervised(G,MRP_iter,gamma_iterations,validation_split,verbose=False,debug=False):
    # Run SD-MRP, but tune gamma automatically by sub-sampling known
    # measurements
    
    MRP = initialise_MRP(G)
    
    # Not very elegant but it works
    best_discount = None
    best_loss = 9999999999999
    
    # Search for best global discount
    for i in range(0,gamma_iterations):
        local_MRP = MRP.copy()
        discount = np.random.uniform() # ranges between 0 and 1, random search should be fine
        
        # Subsampling
        validation_list = []
        for n in local_MRP.nodes(data=True):
            if(not(n[1]['hidden'])):
                if(np.random.uniform() < validation_split):
                    nx.set_node_attributes(local_MRP,{n[0]:True},'hidden')
                    validation_list.append(n[0])
        
        W = initialise_W(local_MRP,default_weight=discount)
        local_MRP = run_MRP(local_MRP,W,MRP_iter)
        
        # Compute and process errors
        pred = []
        labels = []
        for n_id in validation_list:
            n = local_MRP.nodes(data=True)[n_id]
            pred.append(n['E'])
            labels.append(n['label'])
            
        total_error = mean_absolute_error(labels,pred)
        if(total_error < best_loss):
            best_loss = total_error
            best_discount = discount
        
            
    # Run SD-MRP normally with best found discount
    W = initialise_W(MRP,default_weight=best_discount)
    

    # Run MRP
    MRP = run_MRP(MRP,W,MRP_iter,debug=False)

    # Compute error
    pred,labels = get_validation_vectors(MRP)
    total_error = mean_absolute_error(labels,pred) 
    if(verbose):
        print("Loss: " + str(total_error))


    if(debug):
        print(total_error)
        
    return(MRP,total_error,pred)
    
    
    
    
def optimise_weights(G,epochs,MRP_iter,mutation_intensity):
    # Run O-MRP

    # Use CMA-ES for optimisation
    num_params = len(G.edges)
    xopt, es = cma.fmin2(cma_es_obj, len(G.edges) * [0], mutation_intensity,
        options={'maxiter':epochs},args=(G,MRP_iter))
    
    # Convert solution to dictionary format (which we use)
    optimised_W = {}
    i = 0
    for n1,n2,w in G.edges(data=True):
        optimised_W[(n1,n2)] = xopt[i]
        i += 1
    
    # Run MRP
    MRP = initialise_MRP(G)
    MRP = run_MRP(MRP,optimised_W,MRP_iter)
    
    pred,labels = get_validation_vectors(MRP)
    opt_loss = mean_absolute_error(labels,pred)

    return(MRP,opt_loss,optimised_W)
    
    
def cma_es_obj(x, *args):
    # Objective function for O-MRP. Assigns weights from the candidate
    # solution to the edges and runs MRP with those

    MRP = initialise_MRP(args[0])
    W = {}
    i = 0
    for n1,n2,w in MRP.edges(data=True):
        W[(n1,n2)] = x[i]
        i += 1
    MRP = run_MRP(MRP,W,args[1])
    pred,labels = get_validation_vectors(MRP)
    total_error = mean_absolute_error(labels,pred)
    

    return(total_error)
    
    
    
    
def supervise_by_optimisation_auto(G,MRP_iter,optimisation_epochs,mutation_intensity,train_proportion,autoskl_params,average_loss):
    # Run WP-MRP.
    # Optimise to get best weights, then train ML model to predict optimised weights
    # as ground truth. This function is the auto-sklearn version we used in the paper;
    # while we also had code for manually specified models, we did not include it here
    # since it was not used in the paper.
    
    num_params = len(G.edges)
    xopt, es = cma.fmin2(cma_es_obj, len(G.edges) * [0], mutation_intensity,
        options={'maxiter':optimisation_epochs},args=(G,MRP_iter))
    
    optimised_W = {}
    i = 0
    for n1,n2,w in G.edges(data=True):
        optimised_W[(n1,n2)] = xopt[i]
        i += 1
    
    
    MRP = initialise_MRP(G)
    MRP = run_MRP(MRP,optimised_W,MRP_iter)
    
    pred,labels = get_validation_vectors(MRP)
    opt_loss = mean_absolute_error(labels,pred)
    
    
    # Setup feature matrix and label vector
    num_edges = len(G.edges)
    num_attrs = -1
    for n in G.nodes(data=True):
        num_attrs = len(n[1]['A'])
        break
        
    num_features = num_attrs * 2
    
    y = np.zeros(num_edges)
    X = np.zeros((num_edges,num_features))
    
    # Iterate over edges
    
    i = 0
    for n1,n2,a in G.edges(data=True):
        w = optimised_W[(n1,n2)]
        f1 = G.nodes(data=True)[n1]['A']
        f2 = G.nodes(data=True)[n2]['A']
        f = np.concatenate((f1,f2))
        
        # Set features
        X[i,:] = f
        # Set label
        y[i] = w
        
        i += 1
        
    # Split train/test sets
    
    X_train,X_test,y_train,y_test = simple_train_test_split(X,y,train_proportion)
    
    # Train model

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
    
    # Compute error
    
    pred = automl.predict(X_test)
    loss = mean_absolute_error(y_test,pred)
    
    # Return model and loss
    
    return(automl,loss,opt_loss)
    
def train_WP(G,MRP_iter,train_proportion,model,autoskl_params=None,average_loss=True):
    
    true_W = {}
    num_viable = 0
    
    for n1,n2,w in G.edges(data=True):
        if(not(G.nodes(data=True)[n1]['hidden'] or G.nodes(data=True)[n2]['hidden'])):
            l1 = G.nodes(data=True)[n1]['label']
            l2 = G.nodes(data=True)[n2]['label']
            true_weight = l2/max(0.01,l1)
            true_W[(n1,n2)] = true_weight
            num_viable += 1
    
    
    # Setup feature matrix and label vector
    num_edges = len(G.edges)
        
    num_features = 1
    for n in G.nodes(data=True):
        num_features = len(n[1]['A']) * 2
        break
    
    y = np.zeros(num_viable)
    X = np.zeros((num_viable,num_features))
    
    # Iterate over edges
    
    i = 0
    for n1,n2,a in G.edges(data=True):
        if(not(G.nodes(data=True)[n1]['hidden'] or G.nodes(data=True)[n2]['hidden'])):
            w = true_W[(n1,n2)]
            f1 = G.nodes(data=True)[n1]['A']
            f2 = G.nodes(data=True)[n2]['A']
            f = np.concatenate((f1,f2))

            
            # Set features
            X[i,:] = f
            # Set label
            y[i] = w
            
            i += 1

    # Split train/test sets
    
    X_train,X_test,y_train,y_test = simple_train_test_split(X,y,train_proportion)
    
    # Train model


    if(autoskl_params!=None):
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

    else:
        model.fit(X_train,y_train)
        automl = model # Not actually autoML, but this gets the job done
    
    # Compute error
    
    pred = automl.predict(X_test)
    loss = mean_absolute_error(y_test,pred)
    
    # Return model and loss
    
    return(automl,loss)
    
def run_MRP_with_model(G,model,MRP_iter):
    # Predict weights using supplied model, then
    # call MRP function
    
    # Initialise MRP
    MRP = initialise_MRP(G)
    
    W = {}
    
    # Predict weights
    for n1,n2,a in MRP.edges(data=True):
        f1 = G.nodes(data=True)[n1]['A']
        f2 = G.nodes(data=True)[n2]['A']
        f = np.concatenate((f1,f2))
        f = f.reshape(1,len(f))
        
        pred = model.predict(f)[0]
        W[(n1,n2)] = pred
    
    # Run MRP
    
    MRP = run_MRP(MRP,W,MRP_iter)
    
    # Compute loss
    
    pred,labels = get_validation_vectors(MRP)
    loss = mean_absolute_error(labels,pred) # TODO: make variable
    
    # Return loss
    
    return(MRP,loss,pred)

