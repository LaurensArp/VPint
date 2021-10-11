import os
import numpy as np
from VPint.utils.generate_spatial_data import *



def generate_3D_data(user_params=None,generate_features=False):
    
    # Read default parameters
    params = {}
    rel_path = os.path.join(os.path.dirname(__file__),default_params_file)
    df = pd.read_csv(rel_path)
    for i,r in df.iterrows():
        name = r['name']
        dtype = r['type']
        raw_val = r['value']
        if(dtype == 'boolean'):
            val = bool(raw_val)
        elif(dtype == 'float'):
            val = float(raw_val)
        else:
            val = int(raw_val)
        params[name] = val
        
    # Read user parameters
    if(user_params != None):
        for k,v in user_params.items():
            params[k] = v
            
    height = params["param_grid_height"]
    width = params["param_grid_width"]
    depth = params["param_grid_depth"]
    
    # Generate values
    data = np.zeros((height,width,depth))
    for t in range(0,depth):
        data[:,:,t] = generate_data(params)
        
    data_new = data.copy()
    
    t_ac = params["temporal_autocorr"]
    for i in range(0,height):
        for j in range(0,width):
            for i in range(0,depth):
                if(t <= 0):
                    neighbour = data[i,j,t+1]
                    data_new[i,j,t] = (1-t_ac) * data[i,j,t] + t_ac * neighbour
                elif(t >= depth-1):
                    neighbour = data[i,j,t-1]
                    data_new[i,j,t] = (1-t_ac) * data[i,j,t] + t_ac * neighbour
                else:
                    mean_neighbour = np.mean([data[i,j,t-1],data[i,j,t+1]])
                    data_new[i,j,t] = (1-t_ac) * data[i,j,t] + t_ac * mean_neighbour
        
        
    # Generate features
    if(generate_features):
        mean_grid = np.mean(data,axis=0)
        f_grid = assign_features(data,params)
        return(data_new,f_grid)
    
    else:
        return(data_new)
        
    
    