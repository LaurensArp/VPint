import numpy as np
import matplotlib.pyplot as plt

import rasterio

from VPint.WP_MRP import WP_SMRP

def tiff_to_numpy(path):
    """Load GeoTIFF image as numpy array"""
    with rasterio.open(path) as fp:
        vals = np.moveaxis(fp.read().astype(np.float64),0,-1)
        vals = np.nan_to_num(vals,nan=np.nanmean(vals))
    return(vals)

def multiband_VPint(target_img,feature_img,iterations=-1,method='exact',max_gamma=np.inf,min_gamma=0, prioritise_identity=True,priority_intensity=1):
    """Function to run VPint (WP-MRP) on all bands independently"""
    pred_img = target_img.copy()
    for b in range(target_img.shape[2]):
        MRP = WP_SMRP(target_img[:,:,b],feature_img[:,:,b],max_gamma=max_gamma,min_gamma=min_gamma)
        pred_img[:,:,b] = MRP.run(iterations=iterations,method=method)
    return(pred_img)

def apply_cloud_mask(target_img,mask,threshold=None):
    """Apply cloud mask to all bands in the target image"""
    cloud_img = target_img.copy()
    for i in range(0,target_img.shape[0]):
        for j in range(0,target_img.shape[1]):
            if(threshold == None):
                if(np.isnan(mask[i,j,0])):
                    v = np.ones(target_img.shape[2]) * np.nan
                    cloud_img[i,j,:] = v
            else:
                if(mask[i,j,0] > threshold):
                    v = np.ones(target_img.shape[2]) * np.nan
                    cloud_img[i,j,:] = v
    return(cloud_img)

def rgb_composite(inp,max_val=-1,title="",brightness=1.0,rgb=[3,2,1]):
    """Plot an RGB composite of a multi-band optical image"""
        
    img = np.zeros((inp.shape[0],inp.shape[1],3))
    img[:,:,0] = inp[:,:,rgb[0]]
    img[:,:,1] = inp[:,:,rgb[1]]
    img[:,:,2] = inp[:,:,rgb[2]]
    
    if(max_val == -1):
        max_val = np.nanmax(inp)
    
    img[:,:,0] = img[:,:,0] / max_val * 255
    img[:,:,1] = img[:,:,1] / max_val * 255
    img[:,:,2] = img[:,:,2] / max_val * 255
    
    img = np.clip(img*brightness,0,255)
    
    fig = plt.figure(figsize = (7,7))
    ax = fig.add_subplot(111)
    ax.imshow(img.astype(np.uint8))
    ax.set_title(title)
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    plt.show()