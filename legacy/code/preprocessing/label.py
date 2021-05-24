
import rasterio as rio
import rasterio.mask
from rasterio.plot import plotting_extent
import numpy as np
from shapely.geometry import Polygon, Point
import earthpy.spatial as es


   
def get_label_at_xy(path,x,y,label_band=1):
    # Get the label (of a raster dataset) at specified coordinates.
    # x is lon, y is lat
    
    val = -1
    with rio.open(path) as raster:
        row, col = raster.index(x, y)
        val = raster.read(label_band,window=rio.windows.Window(col,row,1,1)) # col first for this function
    
    return(val)
    
    

