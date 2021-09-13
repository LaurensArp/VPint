from pykrige.ok3d import OrdinaryKriging3D
from pykrige.uk3d import UniversalKriging3D

import numpy as np

def ordinary_kriging(grid,variogram_model):
    height = grid.shape[0]
    width = grid.shape[1]
    depth = grid.shape[2]

    gridx = np.arange(0.0, float(width), 1.0)
    gridy = np.arange(0.0, float(height), 1.0)
    gridz = np.arange(0.0, float(depth), 1.0)

    data_size = (~np.isnan(grid)).sum()

    data = np.zeros((data_size,4))
    c = 0
    for i in range(0,height):
        for j in range(0,width):
            for t in range(0,depth):
                if(not(np.isnan(grid[i,j,t]))):
                    data[c,:] = [i,j,t,grid[i,j,t]] 
                    c += 1
                
    OK = OrdinaryKriging3D(data[:,1],data[:,0],data[:,2],data[:,3],variogram_model=variogram_model) # pykrige wants xyz
    kriged_grid, var_grid = OK.execute('grid',gridx,gridy,gridz)
    
    kriged_grid = np.swapaxes(kriged_grid,0,2)
    
    return(kriged_grid, var_grid)


def universal_kriging(grid,variogram_model):
    height = grid.shape[0]
    width = grid.shape[1]
    depth = grid.shape[2]

    gridx = np.arange(0.0, float(width), 1.0)
    gridy = np.arange(0.0, float(height), 1.0)
    gridz = np.arange(0.0, float(depth), 1.0)

    data_size = (~np.isnan(grid)).sum()

    data = np.zeros((data_size,4))
    c = 0
    for i in range(0,height):
        for j in range(0,width):
            for t in range(0,depth):
                if(not(np.isnan(grid[i,j,t]))):
                    data[c,:] = [i,j,t,grid[i,j,t]]
                    c += 1
                    
    OK = UniversalKriging3D(data[:,1],data[:,0],data[:,2],data[:,3],variogram_model=variogram_model) # pykrige wants xyz
    kriged_grid, var_grid = OK.execute('grid',gridx,gridy,gridz)
    
    kriged_grid = np.swapaxes(kriged_grid,0,2) # For some reason it swaps the axes. Verified that it just puts
                                               # the temporal dimension first (similar pictures if indexed this way)
    
    return(kriged_grid, var_grid)


def regression_train(grid,f_grid,model):
    height = grid.shape[0]
    width = grid.shape[1]
    depth = grid.shape[2]

    training_size = (~np.isnan(grid)).sum()
    num_features = f_grid.shape[2]

    X_train = np.zeros((training_size,num_features))
    y_train = np.zeros((training_size))

    c = 0
    for i in range(0,height):
        for j in range(0,width):
            for t in range(0,depth):
                if(not(np.isnan(grid[i,j,t]))):
                    X_train[c,:] = f_grid[i,j,:]
                    y_train[c] = grid[i,j,t]
                    c += 1
    
    model.fit(X_train, y_train)
    return(model)

def regression_run(grid,f_grid,model):
    height = grid.shape[0]
    width = grid.shape[1]
    depth = grid.shape[2]

    pred_grid = grid.copy()

    for i in range(0,height):
        for j in range(0,width):
            for t in range(0,depth):
                if(np.isnan(grid[i,j,t])):
                    f = f_grid[i,j,:]
                    f = f.reshape((1,len(f)))
                    pred = model.predict(f)
                    pred_grid[i,j,t] = pred[0]
    
    return(pred_grid)


def spatial_lag_val(grid,i,j,t,mean):
    h = grid.shape[0] - 1
    w = grid.shape[1] - 1
    d = grid.shape[2] - 1
    
    vec = np.ones(6) * mean
    if(i > 0):
        # Top
        val = grid[i-1,j,t]
        if(not(np.isnan(val))):
            vec[0] = val
    if(j < w):
        # Right
        val = grid[i,j+1,t]
        if(not(np.isnan(val))):
            vec[1] = val
    if(i < h):
        # Bottom
        val = grid[i+1,j,t]
        if(not(np.isnan(val))):
            vec[2] = val
    if(j > 0):
        # Left
        val = grid[i,j-1,t]
        if(not(np.isnan(val))):
            vec[3] = val
    if(t > 0):
        # Earlier
        val = grid[i,j,t-1]
        if(not(np.isnan(val))):
            vec[4] = val
    if(t < d):
        # Later
        val = grid[i,j,t+1]
        if(not(np.isnan(val))):
            vec[5] = val
            
    return(vec)


def SAR_train(grid,f_grid,model):

    # Compute nanmean of grid for mean imputation
    mean_val = np.nanmean(grid)

    # Initialise X_train
    height = grid.shape[0]
    width = grid.shape[1]
    depth = grid.shape[2]

    training_size = (~np.isnan(grid)).sum()
    num_features = f_grid.shape[2] + 6

    X_train = np.zeros((training_size,num_features))
    y_train = np.zeros((training_size))

    c = 0
    for i in range(0,height):
        for j in range(0,width):
            for t in range(0,depth):
                if(not(np.isnan(grid[i,j,t]))):
                    f1 = f_grid[i,j,:]
                    f2 = spatial_lag_val(grid,i,j,t,mean_val)
                    X_train[c,0:-6] = f1
                    X_train[c,-6:] = f2
                    y_train[c] = grid[i,j,t]
                    c += 1
    
    model.fit(X_train, y_train)
    
    return(model)
    
    
def SAR_run(grid,f_grid,model):
    height = grid.shape[0]
    width = grid.shape[1]
    depth = grid.shape[2]
    
    mean_val = np.nanmean(grid)

    pred_grid = grid.copy()

    for i in range(0,height):
        for j in range(0,width):
            for t in range(0,depth):
                if(np.isnan(grid[i,j,t])):
                    f1 = f_grid[i,j,:]
                    f2 = spatial_lag_val(grid,i,j,t,mean_val)
                    f = np.zeros((1,len(f1)+len(f2)))
                    f[0,0:-6] = f1
                    f[0,-6:] = f2
                    pred = model.predict(f)
                    pred_grid[i,j] = pred[0]
    
    return(pred_grid)




def spatial_lag_error(grid,f_grid,i,j,t,mean):
    h = grid.shape[0] - 1
    w = grid.shape[1] - 1
    d = grid.shape[2] - 1
    
    vec = np.ones(6) * mean
    if(i > 0):
        # Top
        val = grid[i-1,j,t]
        if(not(np.isnan(val))):
            vec[0] = val
    if(j < w):
        # Right
        val = grid[i,j+1,t]
        if(not(np.isnan(val))):
            vec[1] = val
    if(i < h):
        # Bottom
        val = grid[i+1,j,t]
        if(not(np.isnan(val))):
            vec[2] = val
    if(j > 0):
        # Left
        val = grid[i,j-1,t]
        if(not(np.isnan(val))):
            vec[3] = val
    if(t > 0):
        # Earlier
        val = grid[i,j,t-1]
        if(not(np.isnan(val))):
            vec[4] = val
    if(t < d):
        # Later
        val = grid[i,j,t+1]
        if(not(np.isnan(val))):
            vec[5] = val
            
    return(vec)


def MA_train(grid,f_grid,model,sub_model):
    
    sub_model = regression_train(grid,f_grid,sub_model)
    sub_pred_grid = regression_run(grid,f_grid,sub_model)
    
    sub_error_grid = sub_pred_grid - grid
    
    mean_error = np.nanmean(sub_error_grid)

    # Initialise X_train
    height = grid.shape[0]
    width = grid.shape[1]
    depth = grid.shape[2]

    training_size = (~np.isnan(grid)).sum()
    num_features = f_grid.shape[2] + 6

    X_train = np.zeros((training_size,num_features))
    y_train = np.zeros((training_size))

    c = 0
    for i in range(0,height):
        for j in range(0,width):
            for t in range(0,depth):
                if(not(np.isnan(grid[i,j,t]))):
                    f1 = f_grid[i,j,:]
                    f2 = spatial_lag_error(sub_error_grid,f_grid,i,j,t,mean_error)
                    X_train[c,0:-6] = f1
                    X_train[c,-6:] = f2
                    y_train[c] = grid[i,j,t]
                    c += 1
    
    model.fit(X_train, y_train)
    
    return(model,sub_model,sub_error_grid)
    
    
def MA_run(grid,f_grid,model,sub_model,sub_error_grid):
    height = grid.shape[0]
    width = grid.shape[1]
    depth = grid.shape[2]
    
    mean_error = np.nanmean(sub_error_grid)

    pred_grid = grid.copy()

    for i in range(0,height):
        for j in range(0,width):
            for t in range(0,depth):
                if(np.isnan(grid[i,j,t])):
                    f1 = f_grid[i,j,:]
                    f2 = spatial_lag_error(sub_error_grid,f_grid,i,j,t,mean_error)
                    f = np.zeros((1,len(f1)+len(f2)))
                    f[0,0:-6] = f1
                    f[0,-6:] = f2
                    pred = model.predict(f)
                    pred_grid[i,j,t] = pred[0]
    
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
    depth = grid.shape[2]

    training_size = (~np.isnan(grid)).sum()
    num_features = f_grid.shape[2] + 12

    X_train = np.zeros((training_size,num_features))
    y_train = np.zeros((training_size))

    c = 0
    for i in range(0,height):
        for j in range(0,width):
            for t in range(0,depth):
                if(not(np.isnan(grid[i,j,t]))):
                    f1 = f_grid[i,j,:]
                    f2 = spatial_lag_error(sub_error_grid,f_grid,i,j,t,mean_error)
                    f3 = spatial_lag_val(grid,i,j,t,mean_val)
                    X_train[c,0:-12] = f1
                    X_train[c,-12:-6] = f2
                    X_train[c,-6:] = f3
                    y_train[c] = grid[i,j,t]
                    c += 1
    
    model.fit(X_train, y_train)
    
    return(model,sub_model,sub_error_grid)
    
    
def ARMA_run(grid,f_grid,model,sub_model,sub_error_grid):
    height = grid.shape[0]
    width = grid.shape[1]
    depth = grid.shape[2]
    
    mean_error = np.nanmean(sub_error_grid)
    mean_val = np.nanmean(grid)

    pred_grid = grid.copy()

    for i in range(0,height):
        for j in range(0,width):
            for t in range(0,depth):
                if(np.isnan(grid[i,j,t])):
                    f1 = f_grid[i,j,:]
                    f2 = spatial_lag_error(sub_error_grid,f_grid,i,j,t,mean_error)
                    f3 = spatial_lag_val(grid,i,j,t,mean_val)
                    f = np.zeros((1,len(f1)+len(f2)+len(f3)))
                    f[0,0:-12] = f1
                    f[0,-12:-6] = f2
                    f[0,-6:] = f3
                    pred = model.predict(f)
                    pred_grid[i,j,t] = pred[0]
    
    return(pred_grid)




def grid_slice(f_grid,i,j,window_height,window_width,fill=None):
    w_h = int(str(window_height/2).split(".")[0]) # Ugly but oh well
    w_w = int(str(window_width/2).split(".")[0])
    
    height = f_grid.shape[0]
    width = f_grid.shape[1]
    num_features = f_grid.shape[2]
    
    if(fill):
        gr_slice = np.ones((window_height,window_width,num_features)) * fill
    else:
        gr_slice = np.zeros((window_height,window_width,num_features))
    
    i2 = w_h
    j2 = w_w
    
    offset_y1 = w_h
    offset_x1 = w_w
    
    offset_y2 = w_h
    offset_x2 = w_w
    
    if((i-w_h) < 0):
        if(fill == None):
            return(False)
        else:
            offset_y1 += (i-w_h)
    elif((i+w_h) > height-1):
        if(fill == None):
            return(False)
        else:
            offset_y2 = (height-1) - i
            
    if((j-w_h) < 0):
        if(fill == None):
            return(False)
        else:
            offset_x1 += (j-w_w)
    elif((j+w_h) > height-1):
        if(fill == None):
            return(False)
        else:
            offset_x2 =  (width) - j
    
    gr_slice[i2-offset_y1:i2+offset_y2,j2-offset_x1:j2+offset_x2] = f_grid[i-offset_y1:i+offset_y2,j-offset_x1:j+offset_x2]

    return(gr_slice)


def CNN_train_pixel(grid,f_grid,model,max_trials=100,epochs=100,validation_split=0.2,name="pixel",window_height=5,window_width=5,fill=False,train_fill=False):
    mean_val = np.nanmean(grid)
    
    training_size = (~np.isnan(grid)).sum()
    num_features = f_grid.shape[2]

    X_train = np.zeros((training_size,window_height,window_width,num_features))
    y_train = np.zeros((training_size))
    
    height = grid.shape[0]
    width = grid.shape[1]
    depth = grid.shape[2]
    
    if(train_fill):
        fill = mean_val
    else:
        fill = False
    
    c = 0
    for i in range(0,height):
        for j in range(0,width):
            for t in range(0,depth):
                if(not(np.isnan(grid[i,j,t]))):
                    gr_slice = grid_slice(f_grid,i,j,window_height,window_width,fill=fill)
                    if(type(gr_slice) != bool):
                        X_train[c,:,:,:] = gr_slice
                        y_train[c] = grid[i,j,t]
                        c += 1
                
    model.fit(X_train, y_train, epochs=epochs, verbose=0)
    
    return(model)
                


def CNN_run_pixel(grid,f_grid,model,window_height=5,window_width=5):
    mean_val = np.nanmean(grid)
    
    test_size = (~np.isnan(grid)).sum()
    num_features = f_grid.shape[2]

    X_test = np.zeros((test_size,window_height,window_width,num_features))
    
    height = grid.shape[0]
    width = grid.shape[1]
    depth = grid.shape[2]
    
    pred_grid = grid.copy()
    
    for i in range(0,height):
        for j in range(0,width):
            for t in range(0,depth):
                if(np.isnan(grid[i,j,t])):
                    gr_slice = grid_slice(f_grid,i,j,window_height,window_width,fill=mean_val)
                    f = np.array([gr_slice])
                    pred_grid[i,j,t] = model.predict(f)[0]
                
    
    
    return(pred_grid)





