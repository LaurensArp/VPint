import VPint.utils.baselines_2D
import VPint.utils.baselines_3D

from VPint.utils.hide_spatial_data import hide_values_uniform, hide_values_sim_cloud
from VPint.utils.hide_spatio_temporal_data import hide_values_uniform_3D, hide_values_sim_cloud_3D

from VPint.SD_MRP import SD_SMRP, SD_STMRP
from VPint.WP_MRP import WP_SMRP, WP_STMRP

import time
import numpy as np
import os

from math import log10, sqrt

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
            tt = time.time()
            pred_grid = MRP.run(iterations=params["iterations"],auto_terminate=params["auto_iter"])
            rt = time.time()
        elif(alg == "WP_MRP"):
            MRP = WP_SMRP(grid,f_grid,model=params["model"])
            if(params["method"] == "predict"):
                MRP.train()
            tt = time.time()
            pred_grid = MRP.run(iterations=params["iterations"],auto_terminate=params["auto_iter"],method=params["method"])
            rt = time.time()
        elif(alg == "OK"):
            tt = time.time()
            pred_grid, var_grid = VPint.utils.baselines_2D.ordinary_kriging(grid,params["variogram_model"])
            rt = time.time()
        elif(alg == "UK"):
            tt = time.time()
            pred_grid, var_grid = VPint.utils.baselines_2D.universal_kriging(grid,params["variogram_model"])
            rt = time.time()
        elif(alg == "basic"):
            model = VPint.utils.baselines_2D.regression_train(grid,f_grid,params["model"])
            tt = time.time()
            pred_grid = VPint.utils.baselines_2D.regression_run(grid,f_grid,model)
            rt = time.time()
        elif(alg == "SAR"):
            model = VPint.utils.baselines_2D.SAR_train(grid,f_grid,params["model"])
            tt = time.time()
            pred_grid = VPint.utils.baselines_2D.SAR_run(grid,f_grid,model)
            rt = time.time()
        elif(alg == "MA"):
            model, sub_model, sub_error_grid = VPint.utils.baselines_2D.MA_train(grid,f_grid,params["model"],params["sub_model"])
            tt = time.time()
            pred_grid = VPint.utils.baselines_2D.MA_run(grid,f_grid,model,sub_model,sub_error_grid)
            rt = time.time()
        elif(alg == "ARMA"):
            model, sub_model, sub_error_grid = VPint.utils.baselines_2D.ARMA_train(grid,f_grid,params["model"],params["sub_model"])
            tt = time.time()
            pred_grid = VPint.utils.baselines_2D.ARMA_run(grid,f_grid,model,sub_model,sub_error_grid)
            rt = time.time()
        elif(alg == "CNN"):
            model = VPint.utils.baselines_2D.CNN_train_pixel(grid,f_grid,params["nn_model"],max_trials=params["nn_max_trials"],
                                    epochs=params["nn_epochs"],train_fill=params["nn_train_fill"],
                                   window_height=params["nn_window_height"],
                                    window_width=params["nn_window_width"])
            tt = time.time()
            pred_grid = VPint.utils.baselines_2D.CNN_run_pixel(grid,f_grid,model,window_height=params["nn_window_height"],
                                      window_width=params["nn_window_width"])
            rt = time.time()
        else:
            print("Invalid algorithm")
    
        measures = compute_measures(pred_grid,grid_true,grid)
        runtimes[it] = float(rt-st)
        result_grids[it,:,:] = pred_grid
        #mae = np.mean(np.absolute(pred_grid-grid_true))
        train_time = float(tt-st)
        run_time = float(rt-tt)
        
        if(save):
            save_results(measures,train_time,run_time,params)
        
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
            tt = time.time()
            pred_grid = MRP.run(iterations=params["iterations"],auto_terminate=params["auto_iter"])
            rt = time.time()
        elif(alg == "WP_MRP"):
            MRP = WP_STMRP(grid,f_grid,model_spatial=params["model_spatial"],model_temporal=params["model_temporal"])
            if(params["method"] == "predict"):
                MRP.train()
            tt = time.time()
            pred_grid = MRP.run(iterations=params["iterations"],auto_terminate=params["auto_iter"],method=params["method"])
            rt = time.time()
        elif(alg == "OK"):
            tt = time.time()
            pred_grid, var_grid = VPint.utils.baselines_3D.ordinary_kriging(grid,params["variogram_model"])
            rt = time.time()
        elif(alg == "UK"):
            tt = time.time()
            pred_grid, var_grid = VPint.utils.baselines_3D.universal_kriging(grid,params["variogram_model"])
            rt = time.time()
        elif(alg == "basic"):
            model = VPint.utils.baselines_3D.regression_train(grid,f_grid,params["model"])
            tt = time.time()
            pred_grid = VPint.utils.baselines_3D.regression_run(grid,f_grid,model)
            rt = time.time()
        elif(alg == "SAR"):
            model = VPint.utils.baselines_3D.SAR_train(grid,f_grid,params["model"])
            tt = time.time()
            pred_grid = VPint.utils.baselines_3D.SAR_run(grid,f_grid,model)
            rt = time.time()
        elif(alg == "MA"):
            model, sub_model, sub_error_grid = VPint.utils.baselines_3D.MA_train(grid,f_grid,params["model"],params["sub_model"])
            tt = time.time()
            pred_grid = VPint.utils.baselines_3D.MA_run(grid,f_grid,model,sub_model,sub_error_grid)
            rt = time.time()
        elif(alg == "ARMA"):
            model, sub_model, sub_error_grid = VPint.utils.baselines_3D.ARMA_train(grid,f_grid,params["model"],params["sub_model"])
            tt = time.time()
            pred_grid = VPint.utils.baselines_3D.ARMA_run(grid,f_grid,model,sub_model,sub_error_grid)
            rt = time.time()
        elif(alg == "CNN"):
            model = VPint.utils.baselines_3D.CNN_train_pixel(grid,f_grid,params["nn_model"],max_trials=params["nn_max_trials"],
                                    epochs=params["nn_epochs"],train_fill=params["nn_train_fill"],
                                   window_height=params["nn_window_height"],
                                    window_width=params["nn_window_width"])
            tt = time.time()
            pred_grid = VPint.utils.baselines_3D.CNN_run_pixel(grid,f_grid,model,window_height=params["nn_window_height"],
                                      window_width=params["nn_window_width"])
            rt = time.time()
        else:
            print("Invalid algorithm")
    
        measures = compute_measures(pred_grid,grid_true,grid)
        et = time.time()
        runtimes[it] = float(rt-st)
        result_grids[it,:,:,:] = pred_grid
        #mae = np.mean(np.absolute(pred_grid-grid_true))
        train_time = float(tt-st)
        run_time = float(rt-tt)
        
        if(save):
            save_results(measures,train_time,run_time,params)
        
    return(result_grids,runtimes)


def compute_measures(pred,true,mask):
    
    # MAE
    diff = np.absolute(true-pred)

    flattened_mask = mask.copy().reshape((np.prod(mask.shape)))
    flattened_diff = diff.reshape((np.prod(diff.shape)))[np.isnan(flattened_mask)]

    mae = np.nanmean(flattened_diff)
    
    # RMSE
    diff = true-pred

    flattened_mask = mask.copy().reshape((np.prod(mask.shape)))
    flattened_diff = diff.reshape((np.prod(diff.shape)))[np.isnan(flattened_mask)]

    rmse = np.mean(np.square(flattened_diff))

    # PSNR
    # Based on https://www.geeksforgeeks.org/python-peak-signal-to-noise-ratio-psnr/
    flattened_mask = mask.copy().reshape((np.prod(mask.shape)))
    flattened_true = true.reshape((np.prod(true.shape)))[np.isnan(flattened_mask)]
    flattened_pred = pred.reshape((np.prod(pred.shape)))[np.isnan(flattened_mask)]

    mse2 = np.nanmean((flattened_true - flattened_pred) ** 2) + 0.001 # 0.001 for smoothing

    if(mse2 == 0):
        return(1)
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse2)) / 100 # /100 because I want 0-1

    # SSIM
    flattened_mask = mask.copy().reshape((np.prod(mask.shape)))
    flattened_true = true.reshape((np.prod(true.shape)))[np.isnan(flattened_mask)]
    flattened_pred = pred.reshape((np.prod(pred.shape)))[np.isnan(flattened_mask)]

    try:
        from skimage.measure import compare_ssim
        (s,d) = compare_ssim(flattened_true,flattened_pred,full=True)
    except:
        print("Error running compare_ssim. For this functionality, please ensure that scikit-image is installed.")
        s = np.nan

    ssim = s
    
    # Return
    return([mae,rmse,psnr,ssim])


def save_results(measures,train_time,run_time,params):
    save_path = params["save_path"]

    mae = measures[0]
    rmse = measures[1]
    psnr = measures[2]
    ssim = measures[3]
    
    if(not(os.path.exists(save_path))):
        with open(save_path,'w') as fp:
            s = "mae,rmse,psnr,ssim,train_time,run_time\n"
            fp.write(s)
            
    with open(save_path,'a') as fp:
        s = str(mae) + "," + str(rmse) + "," + str(psnr) + "," + str(ssim) + "," + str(train_time) + "," + str(run_time) + "\n"
        fp.write(s)


def save_results_old(mae,train_time,run_time,params):
    save_dir = params["save_dir"]
    save_path = params["save_path"]
    setting_name = params["setting_name"]
    alg = params["alg"]
    hidden_method = params["hidden_method"]
    
    if(not(os.path.exists(save_path + "/" + save_dir))):
        os.mkdir(save_path + "/" + save_dir)
    if(not(os.path.exists(save_path + "/" + save_dir + "/" + setting_name))):
        os.mkdir(save_path + "/" + save_dir + "/" + setting_name)
    if(not(os.path.exists(save_path + "/" + save_dir + "/" + setting_name + "/" + alg + "_" + hidden_method + ".csv"))): 
        with open(save_path + "/" + save_dir + "/" + setting_name + "/" + alg + "_" + hidden_method + ".csv",'w') as fp:
            fp.write("mae,train_time,run_time\n")
            
    with open(save_path + "/" + save_dir + "/" + setting_name + "/" + alg + "_" + hidden_method + ".csv",'a') as fp:
        s = str(mae) + "," + str(train_time) + "," + str(run_time) + "\n"
        fp.write(s)