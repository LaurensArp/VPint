import numpy as np
import rasterio




def load_satellite_data(paths_user,normalise=True,red_band=3,green_band=2,blue_band=1):
    # TODO: arbitrary number of bands
    num_img = len(paths_user)
    
    paths = paths_user.copy() # Because of pop modifying original
    path = paths.pop(0)
    
    img_msi = rasterio.open(path)
    b2 = img_msi.read(1) # blue
    b3 = img_msi.read(2) # green
    b4 = img_msi.read(3) # red
    
    data = np.zeros((num_img,b2.shape[0],b2.shape[1],3))

    if(normalise):
        max_b2 = np.max(b2)
        max_b3 = np.max(b3)
        max_b4 = np.max(b4)
        max_val = max(max_b2,max_b3,max_b4)
        b2 = b2 / max_val * 255
        b3 = b3 / max_val * 255
        b4 = b4 / max_val * 255
    img = np.zeros((b2.shape[0],b2.shape[1],3))
    img[:,:,0] = b4
    img[:,:,1] = b3
    img[:,:,2] = b2
    data[0,:,:,:] = img
    
    c = 1
    if(len(paths) > 0):
        for path in paths:
            img_msi = rasterio.open(path)
            b2 = img_msi.read(1) # blue
            b3 = img_msi.read(2) # green
            b4 = img_msi.read(3) # red

            if(normalise):
                max_b2 = np.max(b2)
                max_b3 = np.max(b3)
                max_b4 = np.max(b4)
                max_val = max(max_b2,max_b3,max_b4)
                b2 = b2 / max_val * 255
                b3 = b3 / max_val * 255
                b4 = b4 / max_val * 255
            img = np.zeros((b2.shape[0],b2.shape[1],3))
            img[:,:,0] = b4
            img[:,:,1] = b3
            img[:,:,2] = b2
            data[c,:,:,:] = img
            c += 1
            
    return(data)


def apply_cloud_mask(target,cloud_mask_path,cloud_threshold=50):
    img_cloud_mask = rasterio.open(cloud_mask_path)
    cloud_mask = img_cloud_mask.read(1)
    
    img = target.copy()

    for i in range(0,cloud_mask.shape[0]):
        for j in range(0,cloud_mask.shape[1]):
            if(cloud_mask[i,j] > 50):
                for band in range(0,img.shape[2]):
                    img[i,j,band] = np.nan
   
    return(img)

def rgb_img(img,blue_ind=0,green_ind=1,red_ind=2):
    blue = img[:,:,2]
    green = img[:,:,1]
    red = img[:,:,0]
    rgb = np.zeros((img.shape[0],img.shape[1],3))
    rgb[:,:,0] = red
    rgb[:,:,1] = green
    rgb[:,:,2] = blue
    return(rgb)


def target_features_split(data,target_index=0,cloud_mask=False):
    target = data[target_index,:,:,:]
    features = data[target_index+1:,:,:,:]
    if(target_index > 0):
        features2 = data[0:target_index,:,:,:]
        features = np.concatenate((features,features2),axis=0)
    num_features = features.shape[0] * features.shape[3]
    feature_grid = np.zeros((features.shape[1],features.shape[2],num_features))
    c = 0
    for i in range(0,features.shape[0]):
        for j in range(0,features.shape[3]):
            feature_grid[:,:,c] = features[i,:,:,j]
            c += 1
            
    if(cloud_mask != False):
        target_cloud = apply_cloud_mask(target,cloud_mask)
        return(target,target_cloud,feature_grid)
    
    else:
        return(target,feature_grid)

def msi_to_grid(data,target_index=0,band_index=0,generate_features=True):
    grid = data[target_index,:,:,band_index]
    
    if(generate_features):
        f_grid = np.zeros((grid.shape[0],grid.shape[1],data.shape[0]-1))
        for t in range(1,data.shape[0]):
            f_grid[:,:,t-1] = data[t,:,:,band_index] # TODO: right now this only works for index of 0

        return(grid,f_grid)
    else:
        return(grid)