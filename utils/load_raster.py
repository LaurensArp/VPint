import rasterio

def raster_to_grid(path,bbox_tl,bbox_br):
    img = rasterio.open(path)
    
    left = bbox_tl[1]
    bottom = bbox_br[0]
    right = bbox_br[1]
    top = bbox_tl[0]

    win = rasterio.windows.from_bounds(left, bottom, right, top, img.transform)

    grid = img.read(1,window=win)
    return(grid.astype(float))