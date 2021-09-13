"""Module for artificially hiding spatial data.
"""

import numpy as np
import pandas as pd

def hide_values_uniform(base_grid,probability):
    """
    Return a new version of the supplied grid where values are
    probabilistically (uniform) hidden.

    :param base_grid: 2D grid of target values
    :param probability: probability to hide an individual cell
    :returns: copy of base_grid with some artificially hidden cells
    """
    height = base_grid.shape[0]
    width = base_grid.shape[1]
    new_grid = np.zeros((height,width))
    for i in range(0,height):
        for j in range(0,width):
            if(np.random.rand() < probability):
                new_grid[i][j] = np.nan
            else:
                new_grid[i][j] = base_grid[i][j]
    return(new_grid)


def hide_values_sim_cloud(base_grid,num_points,radius,points=None,num_traj=10):
    """
    Return a new version of the supplied grid where values are
    hidden by simulating cloud cover (random-walk based).

    :param base_grid: 2D grid of target values
    :param num_points: number of clouds to add
    :param radius: size of artificial clouds
    :param points: optional list of user-supplied (y,x) index pairs for the center point of artificial clouds
    :param num_traj: number of trajectories to simulate cloud radiance from center point
    :returns: copy of base_grid with artificially hidden cells
    """
    
    height = base_grid.shape[0]
    width = base_grid.shape[1]
    if(len(base_grid.shape)>2):
        bands = base_grid.shape[2]
        
    new_grid = base_grid.copy()
    
    for c in range(0,num_points):
        if(points != None):
            y = points[c][0]
            x = points[c][1]
        else:
            y = np.random.randint(low=0,high=height-1) # -1 because 0 indexing
            x = np.random.randint(low=0,high=width-1)
            
        new_grid[y,x] = np.nan
        for t in range(0,num_traj):
            i = y
            j = x
            for b in range(0,radius):
                (i,j) = step(i,j,height-1,width-1)
                if(len(base_grid.shape)>2):
                    for band in range(0,bands):
                        new_grid[i,j,band] = np.nan
                else:
                    new_grid[i,j] = np.nan
                
    return(new_grid)
            
    
def hide_values_clustered_values(base_grid,num_points,radius,observation_probability=1.0,points=None,num_traj=10):
    """
    Return a new version of the supplied grid where values are assumed to have been acquired in clusters of convenient-to-measure locations.

    :param base_grid: 2D grid of target values
    :param num_points: number of clusters to add
    :param radius: size of measurement clusters
    :param observation_probability: probability of a visited neighbour to be a known value
    :param points: optional list of user-supplied (y,x) index pairs for the center point of artificial clusters
    :param num_traj: number of trajectories to simulate radiance from center point
    :returns: copy of base_grid with artificially hidden cells
    """
    
    height = base_grid.shape[0]
    width = base_grid.shape[1]
    new_grid = np.zeros((height,width)) * np.nan
    
    for c in range(0,num_points):
        if(points != None):
            y = points[c][0]
            x = points[c][1]
        else:
            y = np.random.randint(low=0,high=height-1) # -1 because 0 indexing
            x = np.random.randint(low=0,high=width-1)
            
        new_grid[y,x] = np.nan
        for t in range(0,num_traj):
            i = y
            j = x
            for b in range(0,radius):
                (i,j) = step(i,j,height-1,width-1)
                new_grid[i,j] = base_grid[i,j]
                
    return(new_grid)
    
        
def step(i,j,i_max,j_max):
    "Internal function to step to a random neighbour"
    valid = False
    while(not(valid)):
        p = np.random.rand()
        if(p <= 0.25):
            # Up
            i_new = i-1
            j_new = j
        elif(p > 0.25 and p <= 0.5):
            # Left
            i_new = i
            j_new = j-1
        elif(p > 0.5 and p <= 0.75):
            # Down
            i_new = i+1
            j_new = j
        else:
            # Right
            i_new = i
            j_new = j+1
        if(not(i_new < 0 or j_new < 0 or i_new > i_max or j_new > j_max)):
            valid = True
            
    return((i_new,j_new))