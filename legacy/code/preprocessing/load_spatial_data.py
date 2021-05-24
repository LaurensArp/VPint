

import geopandas as gpd
import numpy as np
import os
import gc
import time



def load_spatial_data(shapefile_path,missing_value_method,verbose=False):
    # Loads shapefiles into GeoDataFrame and accounts for missing values

    
    # Get a list of all shapefiles at a directory (ie buildings, roads etc)
    folder = os.fsencode(shapefile_path)
    filenames = []
    for file in os.listdir(folder):
        filename = os.fsdecode(file)
        if filename.endswith( ('.shp') ): 
            filenames.append(filename)



    # Don't run all this stuff if no shapefiles in path
    if(len(filenames) < 1):
        print("WARNING: no shapefiles found.")

    else:
        start_time = time.time()

        # Pop and read first shapefile
        S = gpd.read_file(shapefile_path + "/" + filenames.pop(0))

        # Bit of a data-specific hack, in OSM data buildings type is more informative, 
        # but other shapefiles there is no type, so we need to use fclass. To make other
        # code apply, we are copying the fclass column to type if no type column exists.
        if(not('type' in S.columns)):
            S['type'] = S['fclass']
        if(any(S["type"].isnull())):
            S.at[S["type"].isnull(),'type']=np.nan

        # NaN type filtering first shapefile
        if(missing_value_method == "replace"):
            S.at[S["type"].isnull(),'type']="generic"
        elif(missing_value_method == "drop"):
            S = S.dropna(subset=['type'])
        else:
            print("Invalid missing value method.")


        # If more shapefiles in path, add all of them
        if(len(filenames) > 0):
            for file in filenames:
                # Read shapefile
                S_temp = gpd.read_file(shapefile_path + "/" + file)

                # fclass/type hack
                if(not('type' in S_temp.columns)):
                    S_temp['type'] = S_temp['fclass']

                # NaN type filtering
                if(missing_value_method == "replace"):
                    S.at[S["type"].isnull(),'type'] = "generic"
                elif(missing_value_method == "drop"):
                    S_temp = S_temp.dropna(subset=['type'])
                else:
                    print("Invalid missing value method.")



                # Extend S (filtering before because this is an expensive operation)
                S = S.append(S_temp,ignore_index=True)


                # Clear temp from memory
                del S_temp
                gc.collect()


    if(verbose):
        print("Combined DF length: " + str(len(S)))

    end_time = time.time()
    if(verbose):
        print("Total runtime: " + str(end_time - start_time))
    
    return(S)
    
