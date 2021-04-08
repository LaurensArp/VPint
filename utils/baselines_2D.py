from pykrige.ok import OrdinaryKriging
from pykrige.uk import UniversalKriging

import numpy as np

def ordinary_kriging(grid,variogram_model):
    height = grid.shape[0]
    width = grid.shape[1]

    gridx = np.arange(0.0, float(width), 1.0)
    gridy = np.arange(0.0, float(height), 1.0)

    data_size = (~np.isnan(grid)).sum()

    data = np.zeros((data_size,3))
    c = 0
    for i in range(0,height):
        for j in range(0,width):
            if(not(np.isnan(grid[i,j]))):
                data[c,:] = [i,j,grid[i,j]]
                c += 1
                
    OK = OrdinaryKriging(data[:,0],data[:,1],data[:,2],variogram_model=variogram_model)
    kriged_grid, var_grid = OK.execute('grid',gridx,gridy)
    
    return(kriged_grid, var_grid)


def universal_kriging(grid,variogram_model):
    height = grid.shape[0]
    width = grid.shape[1]

    gridx = np.arange(0.0, float(width), 1.0)
    gridy = np.arange(0.0, float(height), 1.0)
    
    data_size = (~np.isnan(grid)).sum()

    data = np.zeros((data_size,3))
    c = 0
    for i in range(0,height):
        for j in range(0,width):
            if(not(np.isnan(grid[i,j]))):
                data[c,:] = [i,j,grid[i,j]]
                c += 1

    OK = UniversalKriging(data[:,0],data[:,1],data[:,2],variogram_model=variogram_model)
    kriged_grid, var_grid = OK.execute('grid',gridx,gridy)
    
    return(kriged_grid, var_grid)


def regression_train(grid,f_grid,model):
    height = grid.shape[0]
    width = grid.shape[1]

    training_size = (~np.isnan(grid)).sum()
    num_features = f_grid.shape[2]

    X_train = np.zeros((training_size,num_features))
    y_train = np.zeros((training_size))

    c = 0
    for i in range(0,height):
        for j in range(0,width):
            if(not(np.isnan(grid[i,j]))):
                X_train[c,:] = f_grid[i,j,:]
                y_train[c] = grid[i,j]
                c += 1
    
    model.fit(X_train, y_train)
    return(model)

def regression_run(grid,f_grid,model):
    height = grid.shape[0]
    width = grid.shape[1]

    pred_grid = grid.copy()

    for i in range(0,height):
        for j in range(0,width):
            if(np.isnan(grid[i,j])):
                f = f_grid[i,j,:]
                pred = model.predict(f.reshape((len(f),1)))
                pred_grid[i,j] = pred[0]
    
    return(pred_grid)


def spatial_lag_val(grid,i,j,mean):
    h = grid.shape[0] - 1
    w = grid.shape[1] - 1
    
    vec = np.ones(4) * mean
    if(i > 0):
        # Top
        val = grid[i-1,j]
        if(not(np.isnan(val))):
            vec[0] = val
    if(j < w):
        # Right
        val = grid[i,j+1]
        if(not(np.isnan(val))):
            vec[1] = val
    if(i < h):
        # Bottom
        val = grid[i+1,j]
        if(not(np.isnan(val))):
            vec[2] = val
    if(j > 0):
        # Left
        val = grid[i,j-1]
        if(not(np.isnan(val))):
            vec[3] = val
            
    return(vec)


def SAR_train(grid,f_grid,model):

    # Compute nanmean of grid for mean imputation
    mean_val = np.nanmean(grid)

    # Initialise X_train
    height = grid.shape[0]
    width = grid.shape[1]

    training_size = (~np.isnan(grid)).sum()
    num_features = f_grid.shape[2] + 4

    X_train = np.zeros((training_size,num_features))
    y_train = np.zeros((training_size))

    c = 0
    for i in range(0,height):
        for j in range(0,width):
            if(not(np.isnan(grid[i,j]))):
                f1 = f_grid[i,j,:]
                f2 = spatial_lag_val(grid,i,j,mean_val)
                X_train[c,0:-4] = f1
                X_train[c,-4:] = f2
                y_train[c] = grid[i,j]
                c += 1
    
    model.fit(X_train, y_train)
    
    return(model)
    
    
def SAR_run(grid,f_grid,model):
    height = grid.shape[0]
    width = grid.shape[1]
    
    mean_val = np.nanmean(grid)

    pred_grid = grid.copy()

    for i in range(0,height):
        for j in range(0,width):
            if(np.isnan(grid[i,j])):
                f1 = f_grid[i,j,:]
                f2 = spatial_lag_val(grid,i,j,mean_val)
                f = np.zeros((1,len(f1)+len(f2)))
                f[0,0:-4] = f1
                f[0,-4:] = f2
                pred = model.predict(f)
                pred_grid[i,j] = pred[0]
    
    return(pred_grid)



def spatial_lag_error(grid,f_grid,i,j,mean):
    h = grid.shape[0] - 1
    w = grid.shape[1] - 1
    
    vec = np.ones(4) * mean
    if(i > 0):
        # Top
        val = grid[i-1,j]
        if(not(np.isnan(val))):
            vec[0] = val
    if(j < w):
        # Right
        val = grid[i,j+1]
        if(not(np.isnan(val))):
            vec[1] = val
    if(i < h):
        # Bottom
        val = grid[i+1,j]
        if(not(np.isnan(val))):
            vec[2] = val
    if(j > 0):
        # Left
        val = grid[i,j-1]
        if(not(np.isnan(val))):
            vec[3] = val
            
    return(vec)


def MA_train(grid,f_grid,model,sub_model):
    
    sub_model = regression_train(grid,f_grid,sub_model)
    sub_pred_grid = regression_run(grid,f_grid,sub_model)
    
    sub_error_grid = sub_pred_grid - grid
    
    mean_error = np.nanmean(sub_error_grid)

    # Initialise X_train
    height = grid.shape[0]
    width = grid.shape[1]

    training_size = (~np.isnan(grid)).sum()
    num_features = f_grid.shape[2] + 4

    X_train = np.zeros((training_size,num_features))
    y_train = np.zeros((training_size))

    c = 0
    for i in range(0,height):
        for j in range(0,width):
            if(not(np.isnan(grid[i,j]))):
                f1 = f_grid[i,j,:]
                f2 = spatial_lag_error(sub_error_grid,f_grid,i,j,mean_error)
                X_train[c,0:-4] = f1
                X_train[c,-4:] = f2
                y_train[c] = grid[i,j]
                c += 1
    
    model.fit(X_train, y_train)
    
    return(model,sub_model,sub_error_grid)
    
    
def MA_run(grid,f_grid,model,sub_model,sub_error_grid):
    height = grid.shape[0]
    width = grid.shape[1]
    
    mean_error = np.nanmean(sub_error_grid)

    pred_grid = grid.copy()

    for i in range(0,height):
        for j in range(0,width):
            if(np.isnan(grid[i,j])):
                f1 = f_grid[i,j,:]
                f2 = spatial_lag_error(sub_error_grid,f_grid,i,j,mean_error)
                f = np.zeros((1,len(f1)+len(f2)))
                f[0,0:-4] = f1
                f[0,-4:] = f2
                pred = model.predict(f)
                pred_grid[i,j] = pred[0]
    
    return(pred_grid)



def ARMA_train(grid,f_grid,model,sub_model):
    
    sub_model = SAR_train(grid,f_grid,sub_model)
    sub_pred_grid = SAR_run(grid,f_grid,sub_model)
    
    sub_error_grid = sub_pred_grid - grid
    
    mean_error = np.nanmean(sub_error_grid)
    mean_val = np.nanmean(grid)

    # Initialise X_train
    height = grid.shape[0]
    width = grid.shape[1]

    training_size = (~np.isnan(grid)).sum()
    num_features = f_grid.shape[2] + 8

    X_train = np.zeros((training_size,num_features))
    y_train = np.zeros((training_size))

    c = 0
    for i in range(0,height):
        for j in range(0,width):
            if(not(np.isnan(grid[i,j]))):
                f1 = f_grid[i,j,:]
                f2 = spatial_lag_error(sub_error_grid,f_grid,i,j,mean_error)
                f3 = spatial_lag_val(grid,i,j,mean_val)
                X_train[c,0:-8] = f1
                X_train[c,-8:-4] = f2
                X_train[c,-4:] = f3
                y_train[c] = grid[i,j]
                c += 1
    
    model.fit(X_train, y_train)
    
    return(model,sub_model,sub_error_grid)
    
    
def ARMA_run(grid,f_grid,model,sub_model,sub_error_grid):
    height = grid.shape[0]
    width = grid.shape[1]
    
    mean_error = np.nanmean(sub_error_grid)
    mean_val = np.nanmean(grid)

    pred_grid = grid.copy()

    for i in range(0,height):
        for j in range(0,width):
            if(np.isnan(grid[i,j])):
                f1 = f_grid[i,j,:]
                f2 = spatial_lag_error(sub_error_grid,f_grid,i,j,mean_error)
                f3 = spatial_lag_val(grid,i,j,mean_val)
                f = np.zeros((1,len(f1)+len(f2)+len(f3)))
                f[0,0:-8] = f1
                f[0,-8:-4] = f2
                f[0,-4:] = f3
                pred = model.predict(f)
                pred_grid[i,j] = pred[0]
    
    return(pred_grid)










