import numpy as np

def assign_csv_to_grid(df,meta):
    grid = np.zeros((meta["res_y"],meta["res_x"]))
    for i,r in df.iterrows():
        lat = r['latitude']
        lon = r['longitude']

        y_index = int(str((lat - meta["min_lat"]) / meta["step_size_y"]).split(".")[0])
        x_index = int(str((lon - meta["min_lon"]) / meta["step_size_x"]).split(".")[0])

        grid[y_index,x_index] = grid[y_index,x_index] + 1
        
    return(grid)

def filter_bbox_csv(df,bbox_tl,bbox_br):
    df = df[df.latitude >= bbox_br[0]]
    df = df[df.latitude <= bbox_tl[0]]
    df = df[df.longitude <= bbox_br[1]]
    df = df[df.longitude >= bbox_tl[1]]
    return(df)