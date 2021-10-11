from VPint.utils.hide_spatial_data import hide_values_uniform,hide_values_sim_cloud

def hide_values_uniform_3D(grid,hidden_proportion):
    new_grid = grid.copy()
    for i in range(0,grid.shape[2]):
        new_grid[:,:,i] = hide_values_uniform(grid[:,:,i],hidden_proportion)
        
    return(new_grid)

def hide_values_sim_cloud_3D(grid,num_points,radius,num_traj):
    new_grid = grid.copy()
    for i in range(0,grid.shape[2]):
        new_grid[:,:,i] = hide_values_sim_cloud(grid[:,:,i],num_points,radius,
                                                num_traj=num_traj)
        
    return(new_grid)



