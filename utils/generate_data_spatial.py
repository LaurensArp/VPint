import numpy as np

def create_grid(grid_height,grid_width,global_mean,global_std,
                stationary=True,nonstationary_points=None):
    """
    Creates a 2D-grid of specified height and width using N(global_mean,global_std).
    If stationary is True, the grid will be stationary, i.e., the mean and std
    will be constant throughout the grid. If stationary is False, individual points
    will be computed using the mean and std from the list of nonstationary_points.
    
    :param grid_height: height of the grid to be generated
    :param grid_width: width of the grid to be generated
    :param global_mean: global mean for values in the entire grid
    :param global_std: global standard deviation for values in the entire grid
    :param stationary: Boolean denoting whether the grid should be stationary
    :param nonstationary_points: list of nonstationary points, where individual
    points are dictionaries with keys 'coords' = (y,x), 'mean' = local mean, and 
    'std' = local standard deviation.
    :returns: generated grid (no spatial autocorrelation)
    """

    # Non-stationary points: list of dictionaries, where each dictionary contains coords, mean and std
    grid = np.zeros((grid_height,grid_width))
    distribution_grid = np.zeros((grid_height,grid_width,2))
    
    if(stationary):
        # Use same distribution for entire grid
        for i in range(0,grid_height):
            for j in range(0,grid_width):
                distribution_grid[i][j][0] = global_mean
                distribution_grid[i][j][1] = global_std

    else:
        if(not(nonstationary_points)):
            # Doesn't work without specified points
            print("No nonstationary points specified.")
            return(False)
            
        # Add nonstationarity sources at locations
        for point in nonstationary_points:
            y = point['coords'][0]
            x = point['coords'][1]
            local_mean = point['mean']
            local_std = point['std']
            
            distribution_grid[y][x][0] = local_mean
            distribution_grid[y][x][1] = local_std
            
            # Set local distributional statistics using local and global values
            for i in range(0,grid_height):
                for j in range(0,grid_width):
                    # Compute distance to point, normalise to 0-1 range
                    dist = abs(i-y) + abs(j-x)
                    dist_relative = dist / (grid_height + grid_width) # TODO: verify this is correct
                    distribution_grid[i][j][0] = dist_relative * local_mean + (1-dist_relative) * global_mean
                    distribution_grid[i][j][1] = dist_relative * local_std + (1-dist_relative) * global_std
                    
    
    for i in range(0,grid_height):
        for j in range(0,grid_width):
            mean = distribution_grid[i][j][0]
            std = distribution_grid[i][j][1]
            grid[i][j] = np.random.normal(mean,std)
        
    return(grid)

def generate_nonstationary_points(num_points,grid_height,grid_width,min_mean,max_mean):
    """
    Generates random nonstationary points within specified constraints.

    :param num_points: number of nonstationary points to generate
    :param grid_height: height of the target grid
    :param grid_width: width of the target grid
    :param min_mean: minimum local mean value
    :param min_mean: maximum local mean value
    :returns: list of nonstationary point dictionaries
    """
    nonstationary_points = []
    for i in range(0,num_points):
        coords = (np.random.randint(low=0,high=grid_height),np.random.randint(low=0,high=grid_width))
        mean = np.random.uniform(low=min_mean,high=max_mean)
        std = np.random.uniform(low=0,high=mean)
        
        point = {
            "coords":coords,
            "mean":mean,
            "std":std
        }
        nonstationary_points.append(point)
        
    return(nonstationary_points)
    
def find_val(grid,cell_coords,neighbour,h,w):
    """
    Function for the easy access to neighbouring values in a
    grid-based representation.

    :param grid: grid to work with
    :param cell_coords: current cell indices
    :param neighbour: direction of the neighbour ("top", "down", "left" or "right")
    :param h: height of the grid
    :param w: width of the grid
    :returns: returns the value of a neighbour, if it exists, False otherwise
    """
    if(neighbour == "top"):
        i = cell_coords[0] - 1
        j = cell_coords[1]
    elif(neighbour == "down"):
        i = cell_coords[0] + 1
        j = cell_coords[1]
    elif(neighbour == "left"):
        i = cell_coords[0]
        j = cell_coords[1] - 1
    elif(neighbour == "right"):
        i = cell_coords[0]
        j = cell_coords[1] + 1
    else:
        i = None
        j = None
        print("aaaaaah")
        
    if(i >= 0 and i < h):
        if(j >= 0 and j < w):
            return(grid[i][j])
        else:
            return False
    else:
        return False

    
def update_grid(grid,ac_params,grid_height,grid_width,isotropy=True,anisotropy_autocorr=None):
    """
    Apply an update to the specified grid to ensure spatial autocorrelation.

    :param grid: grid to work with
    :param ac_params: dictionary of autocorrelation parameters, should include the keys "iterations" (number of times to call the update function), "autocorrelation" 
    (autocorrelation coefficient), "static" (Boolean denoting whether AC is static
    or not). If static==False, also include "mean" (mean AC) and "std" (AC std).
    :param isotropy: Boolean indicating whether grid should be isotropic or not
    :param anisotropy_autocorr: if isotropy==False, dictionary of directional AC
    coefficients ("top", "down", "left" and "right")
    :returns: grid updated with autocorrelation
    """
    grid2 = grid.copy()
    
    for it in range(0,ac_params["iterations"]):
        for i in range(0,grid_height):
            for j in range(0,grid_width):
                val = grid2[i][j]

                if(isotropy):
                    # Find average neighbouring value
                    neighbours = ["top","down","left","right"]
                    avg_val = 0
                    avg_count = 0
                    for n in neighbours:
                        n_val = find_val(grid2,(i,j),n,grid_height,grid_width)
                        if(n_val):
                            avg_val += n_val
                            avg_count += 1

                    avg_val = avg_val / avg_count

                    # Apply autocorrelation update rule

                    if(ac_params['static']):
                        ac = ac_params['autocorrelation']
                    else:
                        ac = np.random.normal(ac_params['mean'],ac_params['std'])
                    new_val = (1-ac) * val + ac * avg_val
                    grid2[i][j] = new_val
                    
                else:
                    # Make influence proportional to coefficients
                    
                    if(not(anisotropy_autocorr)):
                        print("No coefficients found")
                        return(False)
                    
                    neighbours = ["top","down","left","right"]
                    avg_val = 0                   
                    total_weight = anisotropy_autocorr["top"] + anisotropy_autocorr["down"] + \
                                        anisotropy_autocorr["left"] + anisotropy_autocorr["right"]
                    
                    for n in neighbours:
                        n_val = find_val(grid2,(i,j),n,grid_height,grid_width)
                        if(n_val):
                            weight = anisotropy_autocorr[n] / total_weight
                            avg_val += weight * n_val

                    # Apply autocorrelation update rule

                    new_val = (1-autocorrelation) * val + autocorrelation * avg_val
                    grid2[i][j] = new_val
                        
    return(grid2)
    
def assign_features(base_grid,feature_params):
    """
    Automatically assign features to a grid using a constrained uniform distribution.

    :param base_grid: 2D grid of target values
    :param feature_params: dictionary containing feature generation parameters. Should
    include "num" (number of features to generate/feature vector dimensionality), 
    "max" (maximal value of the uniform distribution), "min" (minimal value of the
    uniform distribution), "correlation" (correlation coefficient between random features
    and the target variable).
    :returns: 3D feature grid
    """

    # Add num_features random features, corresponding to label by factor correlation
    # If correlation=1, equal to passing label as feature. If 0, completely random
    
    num_features = feature_params['num']
    feature_min = feature_params['min']
    feature_max = feature_params['max']
    correlation = feature_params['correlation']
    
    height = base_grid.shape[0]
    width = base_grid.shape[1]
    
    feature_grid = np.zeros((height,width,num_features))
    
    for i in range(0,height):
        for j in range(0,width):
            A = np.zeros(num_features)
            for k in range(0,num_features):
                feature_base = np.random.uniform(low=feature_min,high=feature_max)
                feature = correlation*base_grid[i][j] + (1-correlation)*feature_base
                A[k] = feature
        
            feature_grid[i,j,:] = A
            
    return(feature_grid)
            
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