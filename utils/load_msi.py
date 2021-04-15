import numpy as np
import rasterio




def load_satellite_data(paths_user,normalise=True,red_band=3,green_band=2,blue_band=1):
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
        b2 = b2 / max_b2 * 255
        b3 = b3 / max_b3 * 255
        b4 = b4 / max_b4 * 255
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
                b2 = b2 / max_b2 * 255
                b3 = b3 / max_b3 * 255
                b4 = b4 / max_b4 * 255
            img = np.zeros((b2.shape[0],b2.shape[1],3))
            img[:,:,0] = b4
            img[:,:,1] = b3
            img[:,:,2] = b2
            data[c,:,:,:] = img
            c += 1
            
    return(data)



def msi_to_grid(data,target_index=0,band_index=0,generate_features=True):
    grid = data[target_index,:,:,band_index]
    
    if(generate_features):
        f_grid = np.zeros((grid.shape[0],grid.shape[1],data.shape[0]-1))
        for t in range(1,data.shape[0]):
            f_grid[:,:,t-1] = data[t,:,:,band_index] # TODO: right now this only works for index of 0

        return(grid,f_grid)
    else:
        return(grid)