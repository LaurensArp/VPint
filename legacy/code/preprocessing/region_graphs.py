

import geopandas as gpd
import networkx as nx
import numpy as np

from .label import *
    
def create_super_graph_raw(regions,region_bounds,types,region_size_lat,region_size_lon):
    # This function creates one big graph from all the locations we have.
    # As before, the naming is a little awkward for legacy reasons; at an
    # earlier stage this function was intended to combine all regional graphs
    # into one big graph (super graph). Now, we simply use it to create one
    # big graph from all locations.

    ### Create nodes ###

    G = nx.Graph()
    R_num = 0 # To match bounds and dataframes correctly
    for R in regions:
        # Center coordinates
        R_bounds = region_bounds[R_num]
        R_p1 = R_bounds[0]
        R_p2 = R_bounds[1]
        lon = (R_p1[0] + R_p2[0])/2
        lat = (R_p1[1] + R_p2[1])/2

        # Create attribute vector
        attributes = np.zeros(len(types))
        for i,r in R.iterrows():
            object_type = r['type']
            if(object_type in types):
                j = types.index(object_type)
                attributes[j] += 1

        # Add node to graph
        G.add_node("location_"+str(R_num), lon=lon, lat=lat, A=attributes)

        R_num += 1



    ### Add edges ###

    # Ugly implementation, O(|V|^2)...
    for v1 in G.nodes(data=True):
        for v2 in G.nodes(data=True):
            if((v1[0] != v2[0]) and not(G.has_edge(v1[0],v2[0]))):
                lon1 = v1[1]['lon']
                lon2 = v2[1]['lon']
                lat1 = v1[1]['lat']
                lat2 = v2[1]['lat']

                if(
                  ((abs(lon1 - lon2) <= region_size_lon+(0.01*region_size_lon)) and (lat1 == lat2)) 
                      or                                  # Addition is for rounding errors
                  ((abs(lat1 - lat2) <= region_size_lat+(0.01*region_size_lat)) and (lon1 == lon2))):
                    G.add_edge(v1[0],v2[0])
                    
    return(G)
    
    
def convert_super_G(super_G,S,path,region_params,hidden_proportion,from_grid=True):
    # This function prepares a graph to function as MRP.
    
    location_size_lat = region_params[2]
    location_size_lon = region_params[3]

    b = S.bounds
    S_min_lon = min(b['minx'])
    S_max_lon = max(b['maxx'])
    S_min_lat = min(b['miny'])
    S_max_lat = max(b['maxy'])


    # Determine number of hidden nodes
    hidden_number = int(len(super_G.nodes)*hidden_proportion)


    # Create new (directed) graph
    super_H = nx.DiGraph()

    i = 0
    # It is useful to know the max indices, can't use shape in a graph implementation
    biggest_x = 0 
    biggest_y = 0
    for n in super_G.nodes(data=True):

        # Create copy of node with all necessary attributes
        lon = n[1]['lon']
        lat = n[1]['lat']

        diff = abs(lon - S_min_lon)
        x = int(diff / location_size_lon)
        if(x > biggest_x):
            biggest_x = x
        diff = abs(lat - S_min_lat)
        y = int(diff / location_size_lat)
        if(y > biggest_y):
            biggest_y = y

        # If we use a grid/raster dataset, we add true values at this point.
        # Note: we call them labels in the code, but avoided doing so in the paper
        # to avoid confusion with classification. These "labels" are real-valued
        # numbers
        label = np.nan
        if(from_grid):
            label = get_label_at_xy(path,lon,lat)[0][0]
        else:
            label = 0

        A = n[1]['A']

        # Randomly hide nodes at the specified proportion
        hidden = False
        if(np.random.uniform(0,1) < hidden_proportion):
            hidden = True


        super_H.add_node(n[0],x=x,y=y,lon=lon,lat=lat,A=A,label=label,hidden=hidden)
        
        i += 1



    for n1,n2,w in super_G.edges(data=True):
        # Add two directed edges for every undirected edge
        super_H.add_edge(n1,n2)
        super_H.add_edge(n2,n1)
        
        
    # Assigning labels if not from grid (e.g. the COVID dataset is not
    # grid-based)
    
    if(not(from_grid)):
        # Probably really inefficient. For every row in dataframe, if
        # contains lat and lon, add to nearest node
        df = pd.read_csv(path,sep=",",encoding='utf-8')
        for i,r in df.iterrows():
            lat = r['latitude']
            lon = r['longitude']
            # Check if in range
            if(lat >= S_min_lat and lat <= S_max_lat):
                if(lon >= S_min_lon and lon <= S_max_lon):
                    # Add to nearest node
                    closest_name = ""
                    closest_dist = np.Inf
                    for n in super_H.nodes(data=True):
                        lat_n = n[1]['lat']
                        lon_n = n[1]['lon']
                        
                        dist = (abs(lat-lat_n) + abs(lon-lon_n))
                        if(dist < closest_dist):
                            closest_dist = dist
                            closest_name = n[0]
                    
                    current_count = super_H.nodes(data=True)[closest_name]['label']
                    current_count += 1
                    nx.set_node_attributes(super_H,{closest_name:current_count},'label')
        
    return(super_H,biggest_x+1,biggest_y+1)
    
    
    
    
def shuffle_hidden(G,p):
    # Shuffle around which nodes are hidden, so that experimental performance
    # does not depend on getting lucky with a nice configuration, while also
    # not requiring us to preprocess everything again.
    for n in G.nodes(data=True):
        hidden = False
        rand = np.random.uniform()
        if(rand < p):
            hidden = True
        nx.set_node_attributes(G,{n[0]:hidden},'hidden')
    return(G)
    
    
    
    
    
    
