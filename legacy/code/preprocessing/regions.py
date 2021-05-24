
import geopandas as gpd
import time
from shapely.geometry import Polygon



def compute_centroids(S):
    # We used centroid coordinates for assigning objects to nodes
    centroids = []
    for i,r in S.iterrows():
        s = r['geometry']
        b = s.bounds
        centroid = (((b[0] + b[2])/2),((b[1] + b[3])/2))
        centroids.append(centroid)

    S['centroid'] = centroids
   
    return(S)
    
    
    
def clip_area(S,bl,tr):
    # Let's be merciful on our memory and only run everything on
    # the specified bounding box
    br = (tr[0],bl[1])
    tl = (bl[0],tr[1])
    polygon = Polygon([bl, br, tr, tl, bl])
    S_clipped = gpd.clip(S,polygon)
    return(S_clipped)
    
def is_in_bounds(bounds,point):
    # bounds: ((lon,lat),(lon,lat)), point: (lon,lat)
    # bounds is bottom-left, top-right

    p1 = bounds[0]
    p2 = bounds[1]
    
    x = point[0]
    y = point[1]

    if(x > p1[0] and x < p2[0]):
        if(y > p1[1] and y < p2[1]):
            return(True)
    return(False)
    
    
def compute_region_bounds(S,region_size_lon,region_size_lat):
    # The legacy naming for this function can be confusing; the bounds computed here
    # are actually for individual locations (cells/nodes). Initially it was designed
    # for splitting a large dataset (e.g. country) up into multiple region graphs; instead
    # we now use it to split one regional dataset up into multiple location nodes. In
    # terms of functionality, there is no difference between these tasks.


    # Region bounds
    b = S.bounds
    S_min_lon = min(b['minx'])
    S_max_lon = max(b['maxx'])
    S_min_lat = min(b['miny'])
    S_max_lat = max(b['maxy'])


    # Location bounds
    region_bounds = []

    lon = S_min_lon
    lat = S_min_lat

    while(lon < S_max_lon):
        while(lat < S_max_lat):
            bounds = ((lon,lat),(lon+region_size_lon,lat+region_size_lat))
            region_bounds.append(bounds)

            lat += region_size_lat

        lon += region_size_lon
        lat = S_min_lat
        
    return(region_bounds)
    
    
def assign_objects_to_regions(S,region_bounds,region_min_objects=1,verbose=False):
    # Very important function assigning the correct objects from the GeoDataFrame S
    # to the correct location node. Again, using region-code for location-scale
    # purposes.

    start_time = time.time()

    regions = [] # List of GeoDataFrames

    to_copy = [-1 for i in range(0,len(S))]
    S['to_copy'] = to_copy

    for i,r in S.iterrows():
        # Check which region centroid belatgs to
        c = r['centroid']
        R_num = 0
        for bounds in region_bounds:
            p1 = bounds[0]
            p2 = bounds[1]
            # Check coordinate ranges
            if(c[0] > p1[0] and c[0] < p2[0]):
                if(c[1] > p1[1] and c[1] < p2[1]):
                    # Mark region as copy destination
                    S.at[i,'to_copy'] = R_num

                    break # Can't belong to future regions
            R_num += 1


    # Create dataframes for each location (cell/node)
    i = 0
    for bounds in region_bounds:
        R = S.loc[(S['to_copy'] == i)].copy()
        regions.append(R)
        i += 1

    # Restore original S (drop temp column)
    S = S.drop('to_copy',1)

    # Only include regions with enough geographical objects in them
    region_bounds = [region_bounds[i] for i in range(0,len(region_bounds)) if len(regions[i])>=region_min_objects]
    regions = [R for R in regions if len(R)>=region_min_objects]

    if(verbose):
        print(len(region_bounds))
        print(len(regions))

    end_time = time.time()
    if(verbose):
        print("Runtime: " + str(end_time - start_time))
    
    return(regions,region_bounds)
    
    
