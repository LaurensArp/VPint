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

def get_meta(res_y,res_x,bbox_tl,bbox_br):
    # Compute for later
    
    min_lat = min(bbox_tl[0],bbox_br[0])
    max_lat = max(bbox_tl[0],bbox_br[0])

    min_lon = min(bbox_tl[1],bbox_br[1])
    max_lon = max(bbox_tl[1],bbox_br[1])

    step_size_y = (max_lat - min_lat) / res_y
    step_size_x = (max_lon - min_lon) / res_x
    
    meta = {
        "res_y":res_y,
        "res_x":res_x,
        "min_lat":min_lat,
        "max_lat":max_lat,
        "min_lon":min_lon,
        "max_lon":max_lon,
        "step_size_y":step_size_y,
        "step_size_x":step_size_x
    }

    return(meta)