from shapely.geometry import Polygon
import geopandas as gpd

import gc

import numpy as np
import pandas as pd


def load_spatial_data(spatial_data,bbox_tl,bbox_br,missing_value_method):

    S = gpd.read_file(spatial_data.pop(0))
    S = clip_area(S,bbox_tl,bbox_br)

    if(not('type' in S.columns)):
        S['type'] = S['fclass']
    if(any(S["type"].isnull())):
        S.at[S["type"].isnull(),'type']=np.nan

    # NaN type filtering
    if(missing_value_method == "replace"):
        #S.set_value(S["type"].isnull(),'type',"generic")
        S.at[S["type"].isnull(),'type']="generic"
    elif(missing_value_method == "drop"):
        S = S.dropna(subset=['type'])

    for shp in spatial_data:
        S_temp = gpd.read_file(shp)
        S_temp = clip_area(S_temp,bbox_tl,bbox_br)

        if(not('type' in S_temp.columns)):
            S_temp['type'] = S_temp['fclass']
        if(any(S_temp["type"].isnull())):
            S_temp.at[S_temp["type"].isnull(),'type']=np.nan

        # NaN type filtering
        if(missing_value_method == "replace"):
            S_temp.at[S_temp["type"].isnull(),'type']="generic"
        elif(missing_value_method == "drop"):
            S_temp = S_temp.dropna(subset=['type'])

        S = S.append(S_temp,ignore_index=True)

        del S_temp
        gc.collect()
        
    return(S)


def assign_shapes_to_f_grid(S,meta,type_filter_method,additional_params=None):
    
    if(type_filter_method == "taxonomy"):
        num_features = 6
        
        # Build taxonomy
        taxonomy = {}
        df = pd.read_csv(additional_params["taxonomy_path"],sep="\t",header=None)
        for i,r in df.iterrows():
            original = r[0]
            mapping = r[1]
            taxonomy[original] = mapping
            
    elif(type_filter_method == "top_frequent"):
        num_features = additional_params["num_features"]
        
        type_count = {}
        # Find most frequent types
        
        for i,r in S.iterrows():
            f_type = r['type']
            if(f_type in type_count):
                type_count[f_type] = type_count[f_type] + 1
            else:
                type_count[f_type] = 1
        
        type_order = {}
        sort = sorted(type_count.items(), key=lambda x: x[1], reverse=True)
        for i in range(0,num_features-1):
            type_order[sort[i][0]] = i
       
                

    # Build grid
    
    f_grid = np.zeros((meta["res_y"],meta["res_x"],num_features))
    
    # Fill grid
        
    for i,r in S.iterrows():
        s = r['geometry']
        b = s.bounds
        centroid = (((b[1] + b[3])/2),((b[0] + b[2])/2))

        y_index = int(str((centroid[0] - meta["min_lat"]) / meta["step_size_y"]).split(".")[0])
        x_index = int(str((centroid[1] - meta["min_lon"]) / meta["step_size_x"]).split(".")[0])

        f_type = r['type']
        
        if(type_filter_method == "taxonomy"):
            if(f_type in taxonomy):
                if(f_type == 'transportation'):
                    f_index = 0
                elif(f_type == 'commercial'):
                    f_index = 1
                elif(f_type == 'educational'):
                    f_index = 2
                elif(f_type == 'residential'):
                    f_index = 3
                elif(f_type == 'public'):
                    f_index = 4
                else:
                    f_index = 5

            else:
                f_index = 5
        #else:
        #    f_index = 0
            
        elif(type_filter_method == "top_frequent"):
            if(f_type in type_order):
                f_index = type_order[f_type]
            else:
                f_index = num_features - 1 # last element

        f_grid[y_index,x_index,f_index] = f_grid[y_index,x_index,f_index] + 1
        
    return(f_grid)


def clip_area(S,tl,br):
    bl = (tl[0],br[1])
    tr = (br[0],tl[1])
    
    order = [bl,br,tr,tl,bl]
    points = []
    for coords in order:
        points.append((coords[1],coords[0]))
    
    polygon = Polygon(points)
    S_clipped = gpd.clip(S,polygon)
    return(S_clipped)



def normalise_attributes(f_grid,method):
    height = f_grid.shape[0]
    width = f_grid.shape[1]
    num_features = f_grid.shape[2]
    
    new_grid = np.zeros((height,width,num_features))
    
    # Unit length, fastest, probably worst
    if(method == "unit"):
        for i in range(0,height):
            for j in range(0,width):
                A = f_grid[i,j,:]
                length = sum(A)
                if(length > 0):
                    A = A / sum(A)
                    new_grid[i,j,:] = A
    
    # Z-score normalisation, common in ML, needs 2 passes over all nodes
    elif(method == "z_score"):
        X = f_grid.reshape((height*width,num_features))
            
        means = np.zeros(num_features)
        stds = np.zeros(num_features)
        
        for f in range(0,num_features):
            means[f] = np.mean(X[:,f])
            stds[f] = np.std(X[:,f])
        
        for i in range(0,height):
            for j in range(0,width):
                A = f_grid[i,j,:]
                A_new = np.zeros(num_features)
                for f in range(0,num_features):
                    A_new[f] = (A[f] - means[f]) / stds[f]
                new_grid[i,j,:] = A_new
        
    # Mean normalisation, needs 2 passes
    elif(method == "mean_norm"):
        X = f_grid.reshape((height*width,num_features))
            
        means = np.zeros(num_features)
        maxs = np.zeros(num_features)
        mins = np.zeros(num_features)
        
        for f in range(0,num_features):
            means[f] = np.mean(X[:,f])
            maxs[f] = np.max(X[:,f])
            mins[f] = np.min(X[:,f])
        
        for i in range(0,height):
            for j in range(0,width):
                A = f_grid[i,j,:]
                A_new = np.zeros(num_features)
                for f in range(0,num_features):
                    A_new[f] = (A[f] - means[f]) / max((maxs[f] - mins[f]),0.001)
                new_grid[i,j,:] = A_new
            
    elif(method == "none"):
        new_grid = f_grid.copy()
    else:
        print("Invalid normalisation method.")
        
    return(new_grid)