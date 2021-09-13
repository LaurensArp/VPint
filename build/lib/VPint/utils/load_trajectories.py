import numpy as np
import matplotlib.pyplot as plt


def filter_bbox(df,bbox_tl,bbox_br):

    df = df[df.source_lat >= bbox_br[0]]
    df = df[df.source_lat <= bbox_tl[0]]
    df = df[df.target_lat >= bbox_br[0]]
    df = df[df.target_lat <= bbox_tl[0]]

    df = df[df.source_lon <= bbox_br[1]]
    df = df[df.source_lon >= bbox_tl[1]]
    df = df[df.target_lon <= bbox_br[1]]
    df = df[df.target_lon >= bbox_tl[1]]

    return(df)

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

def assign_traj_to_grid(df,meta,groupby="tod"):
    
    # Create grid
    if(groupby == "tod"):
        num_timesteps = 4
    else:
        num_timesteps = 1
    
    grid = np.zeros((meta["res_y"],meta["res_x"],num_timesteps,2))
    
    # Find appropriate t index
    
    for i,r in df.iterrows():
        if(groupby == "tod"):
            h = int(r['timestamp'].split(" ")[1].split(":")[0]) # ugly but it works
            if(h >= 0 and h < 6):
                #r['time_of_day'] = 'night'
                t_index = 0
            elif(h >= 6 and h < 12):
                #r['time_of_day'] = 'morning'
                t_index = 1
            elif(h >= 12 and h < 18):
                #r['time_of_day'] = 'afternoon'
                t_index = 2
            elif(h >= 18 and h <= 24):
                #r['time_of_day'] = 'evening'
                t_index = 3
        else:
            print("Invalid groupby argument")
            t_index = 0

        # Update origin cell

        source_lat = r['source_lat']
        source_lon = r['source_lon']

        y_index = int(str((source_lat - meta["min_lat"]) / meta["step_size_y"]).split(".")[0])
        x_index = int(str((source_lon - meta["min_lon"]) / meta["step_size_x"]).split(".")[0])

        grid[y_index,x_index,t_index,0] = grid[y_index,x_index,t_index,0] + 1

        # Update destination cell

        target_lat = r['target_lat']
        target_lon = r['target_lon']

        y_index = int(str((target_lat - meta["min_lat"]) / meta["step_size_y"]).split(".")[0])
        x_index = int(str((target_lon - meta["min_lon"]) / meta["step_size_x"]).split(".")[0])

        grid[y_index,x_index,t_index,1] = grid[y_index,x_index,t_index,1] + 1
        
    return(grid)


def traj_rgb_composite(grid,consistent=True):
    
    max_red = np.max(grid[:,:,:,0])
    max_blue = np.max(grid[:,:,:,1])

    for t in range(0,grid.shape[2]):   
        rgb = np.zeros((grid.shape[0],grid.shape[1],3))
        
        if(consistent):
            rgb[:,:,0] = grid[:,:,t,0]/max_red * 255 # source, red
            rgb[:,:,2] = grid[:,:,t,1]/max_blue * 255 # target, blue
        else:
            rgb[:,:,0] = grid[:,:,t,0] # source, red
            rgb[:,:,2] = grid[:,:,t,1] # target, blue

        plt.imshow(rgb)
        plt.title("Timestep " + str(t))
        plt.show()



def transform_file(path,save_path=None):
    df = pd.read_csv(path)

    source_lat = np.zeros(len(df['source_point']))
    source_lon = np.zeros(len(df['source_point']))
    target_lat = np.zeros(len(df['source_point']))
    target_lon = np.zeros(len(df['source_point']))

    # Lat is y, lon is x

    c = 0
    for i,r in df.iterrows():
        p1 = shapely.wkt.loads(r['source_point'])
        p2 = shapely.wkt.loads(r['target_point'])
        source_lat[c] = p1.y
        source_lon[c] = p1.x
        target_lat[c] = p2.y
        target_lon[c] = p2.x  
        c += 1

    df['source_lat'] = source_lat
    df['source_lon'] = source_lon
    df['target_lat'] = target_lat
    df['target_lon'] = target_lon

    df2 = df.drop(['taxi_id','trajectory_id','source_point','target_point'],axis=1)

    if(save_path == None):
        save_path = path.split('.')[0] + "_transformed" + path.split('.')[1]
    df2.to_csv(save_path)

    
    
