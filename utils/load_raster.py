import rasterio

def raster_to_grid(path,bbox_tl,bbox_br):
    img = rasterio.open(path)
    
    left = bbox_tl[0]
    bottom = bbox_br[1]
    right = bbox_br[0]
    top = bbox_tl[1]

    win = rasterio.windows.from_bounds(left, bottom, right, top, img.transform)

    grid = img.read(1,window=win)
    return(grid)