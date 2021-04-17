import utils.baselines_2D
import utils.baselines_3D

from utils.hide_spatial_data import hide_values_uniform, hide_values_sim_cloud
from utils.hide_spatio_temporal_data import hide_values_uniform_3D, hide_values_sim_cloud_3D

from MRPinterpolation.SD_MRP import SD_SMRP, SD_STMRP
from MRPinterpolation.WP_MRP import WP_SMRP, WP_STMRP

import time
import numpy as np
import os

def run_experiments_2D(grid_true,f_grid,alg,iterations,params,hidden_method="random",save=False):

    result_grids = np.zeros((iterations,grid_true.shape[0],grid_true.shape[1]))
    runtimes = np.zeros(iterations)
        
    for it in range(0,iterations):
    
        if(hidden_method == "random"):
            grid = hide_values_uniform(grid_true,params["hidden_proportion"])
        elif(hidden_method == "clouds"):
            grid = hide_values_sim_cloud(grid_true,params["num_points"],params["radius"],
                                         num_traj=params["num_traj"])
        else:
            print("Invalid method")
        
        st = time.time()

        if(alg == "SD_MRP"):
            MRP = SD_SMRP(grid)
            MRP.find_gamma(params["SD_epochs"],params["subsample_proportion"],
                          sub_iterations=params["sub_iterations"])
            pred_grid = MRP.run(params["iterations"])
        elif(alg == "WP_MRP"):
            MRP = WP_SMRP(grid,f_grid,params["model"])
            MRP.train()
            pred_grid = MRP.run(params["iterations"])
        elif(alg == "OK"):
            pred_grid, var_grid = utils.baselines_2D.ordinary_kriging(grid,params["variogram_model"])
        elif(alg == "UK"):
            pred_grid, var_grid = utils.baselines_2D.universal_kriging(grid,params["variogram_model"])
        elif(alg == "basic"):
            model = utils.baselines_2D.regression_train(grid,f_grid,params["model"])
            pred_grid = utils.baselines_2D.regression_run(grid,f_grid,model)
        elif(alg == "SAR"):
            model = utils.baselines_2D.SAR_train(grid,f_grid,params["model"])
            pred_grid = utils.baselines_2D.SAR_run(grid,f_grid,model)
        elif(alg == "MA"):
            model, sub_model, sub_error_grid = utils.baselines_2D.MA_train(grid,f_grid,params["model"],params["sub_model"])
            pred_grid = utils.baselines_2D.MA_run(grid,f_grid,model,sub_model,sub_error_grid)
        elif(alg == "ARMA"):
            model, sub_model, sub_error_grid = utils.baselines_2D.ARMA_train(grid,f_grid,params["model"],params["sub_model"])
            pred_grid = utils.baselines_2D.ARMA_run(grid,f_grid,model,sub_model,sub_error_grid)
        elif(alg == "CNN"):
            model = utils.baselines_2D.CNN_train_pixel(grid,f_grid,params["nn_model"],max_trials=params["nn_max_trials"],
                                    epochs=params["nn_epochs"],train_fill=params["nn_train_fill"],
                                   window_height=params["nn_window_height"],
                                    window_width=params["nn_window_width"])
            pred_grid = utils.baselines_2D.CNN_run_pixel(grid,f_grid,model,window_height=params["nn_window_height"],
                                      window_width=params["nn_window_width"])
        else:
            print("Invalid algorithm")
    
        et = time.time()
        runtimes[it] = int(et-st)
        result_grids[it,:,:] = pred_grid
        mae = np.mean(np.absolute(pred_grid-grid_true))
        runtime = int(et-st)
        
        if(save):
            save_results(mae,runtime,params)
        
    return(result_grids,runtimes)




def run_experiments_3D(grid_true,f_grid,alg,iterations,params,hidden_method="random",save=False):

    result_grids = np.zeros((iterations,grid_true.shape[0],grid_true.shape[1],grid_true.shape[2]))
    runtimes = np.zeros(iterations)
        
    for it in range(0,iterations):
        if(hidden_method == "random"):
            grid = hide_values_uniform_3D(grid_true,params["hidden_proportion"])
        elif(hidden_method == "clouds"):
            grid = hide_values_sim_cloud_3D(grid_true,params["num_points"],params["radius"],
                                         num_traj=params["num_traj"])
        else:
            print("Invalid method")
        
        st = time.time()

        if(alg == "SD_MRP"):
            MRP = SD_STMRP(grid)
            MRP.find_discounts(params["SD_epochs"],params["subsample_proportion"],
                          sub_iterations=params["sub_iterations"])
            pred_grid = MRP.run(params["iterations"])
        elif(alg == "WP_MRP"):
            MRP = WP_STMRP(grid,f_grid,params["model"])
            MRP.train()
            pred_grid = MRP.run(params["iterations"])
        elif(alg == "OK"):
            pred_grid, var_grid = utils.baselines_3D.ordinary_kriging(grid,params["variogram_model"])
        elif(alg == "UK"):
            pred_grid, var_grid = utils.baselines_3D.universal_kriging(grid,params["variogram_model"])
        elif(alg == "basic"):
            model = utils.baselines_3D.regression_train(grid,f_grid,params["model"])
            pred_grid = utils.baselines_3D.regression_run(grid,f_grid,model)
        elif(alg == "SAR"):
            model = utils.baselines_3D.SAR_train(grid,f_grid,params["model"])
            pred_grid = utils.baselines_3D.SAR_run(grid,f_grid,model)
        elif(alg == "MA"):
            model, sub_model, sub_error_grid = utils.baselines_3D.MA_train(grid,f_grid,params["model"],params["sub_model"])
            pred_grid = utils.baselines_3D.MA_run(grid,f_grid,model,sub_model,sub_error_grid)
        elif(alg == "ARMA"):
            model, sub_model, sub_error_grid = utils.baselines_3D.ARMA_train(grid,f_grid,params["model"],params["sub_model"])
            pred_grid = utils.baselines_3D.ARMA_run(grid,f_grid,model,sub_model,sub_error_grid)
        elif(alg == "CNN"):
            model = utils.baselines_3D.CNN_train_pixel(grid,f_grid,params["nn_model"],max_trials=params["nn_max_trials"],
                                    epochs=params["nn_epochs"],train_fill=params["nn_train_fill"],
                                   window_height=params["nn_window_height"],
                                    window_width=params["nn_window_width"])
            pred_grid = utils.baselines_3D.CNN_run_pixel(grid,f_grid,model,window_height=params["nn_window_height"],
                                      window_width=params["nn_window_width"])
        else:
            print("Invalid algorithm")
    
        et = time.time()
        runtimes[it] = int(et-st)
        result_grids[it,:,:,:] = pred_grid
        mae = np.mean(np.absolute(pred_grid-grid_true))
        runtime = int(et-st)
        
        if(save):
            save_results(mae,runtime,params)
        
    return(result_grids,runtimes)



def save_results(mae,runtime,params):
    save_dir = params["save_dir"]
    save_path = params["save_path"]
    setting_name = params["setting_name"]
    alg = params["alg"]
    hidden_method = params["hidden_method"]
    
    if(not(os.path.exists(save_path + "/" + save_dir))):
        os.mkdir(save_path + "/" + save_dir)
    if(not(os.path.exists(save_path + "/" + save_dir + "/" + setting_name))):
        os.mkdir(save_path + "/" + save_dir + "/" + setting_name)
    if(not(os.path.exists(save_path + "/" + save_dir + "/" + setting_name + "/" + alg + ".csv"))): 
        with open(save_path + "/" + save_dir + "/" + setting_name + "/" + alg + "_" + hidden_method + ".csv",'w') as fp:
            fp.write("mae,runtime\n")
            
    with open(save_path + "/" + save_dir + "/" + setting_name + "/" + alg + "_" + hidden_method + ".csv",'a') as fp:
        s = str(mae) + "," + str(runtime) + "\n"
        fp.write(s)