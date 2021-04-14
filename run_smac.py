from utils.generate_spatial_data import *
from utils.hide_spatial_data import *

import smac
from smac.configspace import ConfigurationSpace
from smac.scenario.scenario import Scenario
from ConfigSpace.conditions import InCondition
from ConfigSpace.hyperparameters import CategoricalHyperparameter,UniformFloatHyperparameter, UniformIntegerHyperparameter
#from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.facade.smac_ac_facade import SMAC4AC

from utils.experiments import run_experiments_2D, run_experiments_3D
from sklearn.linear_model import LinearRegression


# Setup

# General

setting_name = "test_run"

bbox_tl = (41.249814,-8.734482)
bbox_br = (41.056908,-8.502107)

res_y = 100
res_x = 100

hidden_method = "random"
alg = "basic"

hidden_proportion = 0.8

num_points = 8
radius =  350
num_traj = 75


# SMAC

max_runtime = 1 # hours
max_memory_per_run = 5 # GB



# Trajectories

traj_path_original = "/mnt/c/Users/Laurens/Projects/Porto Trajectories/porto_trajectories_all.csv"
traj_path = "/mnt/c/Users/Laurens/Projects/Porto Trajectories/porto_trajectories_preprocessed.csv"

num_timesteps = 4


# Features

taxonomy_path = "/mnt/e/User Files/Projects/Master Thesis/Working/type_taxonomy_v1.tsv"

spatial_path = "/mnt/c/Users/Laurens/Projects/Portugal Shapefiles/"
spatial_sources = ["gis_osm_buildings_a_free_1.shp",
                   "gis_osm_natural_free_1.shp",
                  "gis_osm_places_free_1.shp",
                  "gis_osm_pofw_free_1.shp",
                  "gis_osm_traffic_a_free_1.shp",
                  "gis_osm_transport_a_free_1.shp"]

spatial_data = [(spatial_path + s) for s in spatial_sources]







cs = ConfigurationSpace()




# Variable parameters


# Methods

missing_value_method_smac = CategoricalHyperparameter("missing_value_method", ["replace","drop"], default_value="drop")
type_filter_method_smac = CategoricalHyperparameter("type_filter_method", ["top_frequent","taxonomy","none"],
                                                    default_value="taxonomy")
feature_normalisation_method_smac = CategoricalHyperparameter("feature_normalisation_method", ["unit","z_score",
                                                                          "mean_norm","none"],
                                                    default_value="mean_norm")
if(alg != "OK" and  alg != "UK" and alg != "SD_MRP"):
    cs.add_hyperparameters([missing_value_method_smac,type_filter_method_smac,feature_normalisation_method_smac])



# Type parameters

type_top_frequent_smac = UniformIntegerHyperparameter("type_top_frequent", 1, 100, default_value=10)

use_type_top_frequent_smac = InCondition(child=type_top_frequent_smac, parent=type_filter_method_smac,values=["top_frequent"])

if(alg != "OK" and  alg != "UK" and alg != "SD_MRP"):
    cs.add_hyperparameters([type_top_frequent_smac])
    cs.add_conditions([use_type_top_frequent_smac])



# Algorithm specific


if(alg == "SD_MRP"):
    MRP_iterations_smac = UniformIntegerHyperparameter("MRP_iterations", 1, 200, default_value=50)
    MRP_SD_epochs_smac = UniformIntegerHyperparameter("MRP_SD_epochs", 1, 200, default_value=100)
    MRP_sub_iterations_smac = UniformIntegerHyperparameter("MRP_sub_iterations", 1, 200, default_value=100)
    MRP_subsample_proportion_smac = UniformFloatHyperparameter("MRP_subsample_proportion", 0.01, 1.0, default_value=0.5)
    cs.add_hyperparameters([MRP_iterations_smac,MRP_SD_epochs_smac,MRP_sub_iterations_smac,MRP_subsample_proportion_smac])
    
elif(alg == "WP_MRP"):
    MRP_iterations_smac = UniformIntegerHyperparameter("MRP_iterations", 1, 200, default_value=50)
    cs.add_hyperparameters([MRP_iterations_smac])
    
elif(alg == "OK"):
    OK_variogram_smac = CategoricalHyperparameter("OK_variogram", 
                                                  ["linear","power","gaussian","spherical","exponential","hole-effect"], 
                                                  default_value="linear")
    cs.add_hyperparameters([OK_variogram_smac])
elif(alg == "UK"):
    UK_variogram_smac = CategoricalHyperparameter("UK_variogram", 
                                                  ["linear","power","gaussian","spherical","exponential","hole-effect"], 
                                                  default_value="linear")
    cs.add_hyperparameters([UK_variogram_smac])




    
    
    
max_runtime_seconds = int(max_runtime * 60 * 60)
max_memory_usage_mb = max_memory_per_run * 1024


smac_current_id = 1
if(not(os.path.exists("smac/smac_output/id.txt"))): 
    with open("smac/smac_output/id.txt",'w') as fp:
        fp.write("2")
else:
    with open("smac/smac_output/id.txt",'r') as fp:
        s = fp.read()
        smac_current_id = int(s)
    with open("smac/smac_output/id.txt",'w') as fp:
        fp.write(str(smac_current_id+1))
        
#if(not(os.path.exists("smac/smac_output/conditions.csv"))): 
#    with open("smac/smac_output/conditions.csv",'w') as fp:
#        fp.write("id,label_set,method,training_set_name,test_set_name\n")
#with open("smac/smac_output/conditions.csv",'a') as fp:
#    s = str(smac_current_id) + "," + setting_name + "," + alg + "," + training_set_name + "," + test_set_name + "\n"
#    fp.write(s)
    
        

scenario = Scenario({"run_obj": "quality",
                     "output_dir": "smac/smac_output/" + setting_name + "/" + alg + "/" + str(smac_current_id),
                     "wallclock_limit":max_runtime_seconds, 
                     "cs":cs
                    })





def run_config_trajectories(cfg):
    
    cfg = {k: cfg[k] for k in cfg if cfg[k]}
    
    additional_params = {}
    if(cfg["type_filter_method"] == "taxonomy"):
        additional_params["taxonomy_path"] = taxonomy_path
    elif(cfg["type_filter_method"] == "top_frequent"):
        additional_params["num_features"] = cfg["type_top_frequent"]

    # Preprocess
    
    df = pd.read_csv(traj_path)
    df = filter_bbox(df,bbox_tl,bbox_br)
    meta = get_meta(res_y,res_x,bbox_tl,bbox_br)
    grid = assign_traj_to_grid(df,meta)

    meta = get_meta(res_y,res_x,bbox_tl,bbox_br)
    S = load_spatial_data(spatial_data,bbox_tl,bbox_br,cfg["missing_value_method"])
    f_grid = assign_shapes_to_f_grid(S,meta,cfg["type_filter_method"],additional_params=additional_params)
    f_grid = normalise_attributes(f_grid,cfg["feature_normalisation_method"])
    
    # Setup
    
    autoskl_current_id = 1
    if(not(os.path.exists("autosklearn/id.txt"))): 
        with open("autosklearn/id.txt",'w') as fp:
            fp.write("2")
    else:
        with open("autosklearn/id.txt",'r') as fp:
            s = fp.read()
            autoskl_current_id = int(s)
        with open("autosklearn/id.txt",'w') as fp:
            fp.write(str(autoskl_current_id+1))



    model = autosklearn.regression.AutoSklearnRegressor(
        time_left_for_this_task=120,
        per_run_time_limit=30,
        tmp_folder="autosklearn/weights/" + str(autoskl_current_id) + "temp",
        output_folder="/anonymised/path/autosklearn/weights/" + str(autoskl_current_id) + "out",
        delete_tmp_folder_after_terminate=True,
        delete_output_folder_after_terminate=True,
    )

    sub_model = LinearRegression()


    # Algorithms
    if(alg == "OK"):
        params = {
            # Random noise
            "hidden_proportion":hidden_proportion,
            # Cloud noise
            "num_points":num_points,
            "radius":radius,
            "num_traj":num_traj,
            # Model specific
            "variogram_model":cfg["OK_variogram"],
        }
        result_grids, runtimes = run_experiments_3D(grid,f_grid,"OK",num_runs,
                                                        params,hidden_method=hidden_method)
        result = result_grids[0,:,:,:]
        error = np.mean(np.absolute(result-grid))
        return(error)
    
    elif(alg == "UK"):
        params = {
            # Random noise
            "hidden_proportion":hidden_proportion,
            # Cloud noise
            "num_points":num_points,
            "radius":radius,
            "num_traj":num_traj,
            # Model specific
            "variogram_model":cfg["UK_variogram"],
        }
        result_grids, runtimes = run_experiments_3D(grid,f_grid,"UK",num_runs,
                                                        params,hidden_method=hidden_method)
        result = result_grids[0,:,:,:]
        error = np.mean(np.absolute(result-grid))
        return(error)
    
    elif(alg == "basic"):
        params = {
            # Random noise
            "hidden_proportion":hidden_proportion,
            # Cloud noise
            "num_points":num_points,
            "radius":radius,
            "num_traj":num_traj,
            # Model specific
            "model":model,
        }
        result_grids, runtimes = run_experiments_3D(grid,f_grid,"basic",num_runs,
                                                        params,hidden_method=hidden_method)
        result = result_grids[0,:,:,:]
        error = np.mean(np.absolute(result-grid))
        return(error)

    elif(alg == "SAR"):
        params = {
            # Random noise
            "hidden_proportion":hidden_proportion,
            # Cloud noise
            "num_points":num_points,
            "radius":radius,
            "num_traj":num_traj,
            # Model specific
            "model":model,
        }
        result_grids, runtimes = run_experiments_3D(grid,f_grid,"SAR",num_runs,
                                                        params,hidden_method=hidden_method)
        result = result_grids[0,:,:,:]
        error = np.mean(np.absolute(result-grid))
        return(error)
    
    elif(alg == "MA"):
        params = {
            # Random noise
            "hidden_proportion":hidden_proportion,
            # Cloud noise
            "num_points":num_points,
            "radius":radius,
            "num_traj":num_traj,
            # Model specific
            "model":model,
            "sub_model":sub_model,
        }
        result_grids, runtimes = run_experiments_3D(grid,f_grid,"MA",num_runs,
                                                        params,hidden_method=hidden_method)
        result = result_grids[0,:,:,:]
        error = np.mean(np.absolute(result-grid))
        return(error)
        
    elif(alg == "ARMA"):
        params = {
            # Random noise
            "hidden_proportion":hidden_proportion,
            # Cloud noise
            "num_points":num_points,
            "radius":radius,
            "num_traj":num_traj,
            # Model specific
            "model":model,
            "sub_model":sub_model,
        }
        result_grids, runtimes = run_experiments_3D(grid,f_grid,"ARMA",num_runs,
                                                        params,hidden_method=hidden_method)
        result = result_grids[0,:,:,:]
        error = np.mean(np.absolute(result-grid))
        return(error)
        
    elif(alg == "SD_MRP"):
        params = {
            # Random noise
            "hidden_proportion":hidden_proportion,
            # Cloud noise
            "num_points":num_points,
            "radius":radius,
            "num_traj":num_traj,
            # Model specific
            "iterations":cfg["MRP_iterations"],
            "SD_epochs":cfg["MRP_SD_epochs"],
            "subsample_proportion":cfg["MRP_subsample_proportion"],
            "sub_iterations":cfg["MRP_sub_iterations"],
        }
        result_grids, runtimes = run_experiments_3D(grid,f_grid,"SD_MRP",num_runs,
                                                        params,hidden_method=hidden_method)
        result = result_grids[0,:,:,:]
        error = np.mean(np.absolute(result-grid))
        return(error)
        
    elif(alg == "WP_MRP"):
        params = {
            # Random noise
            "hidden_proportion":hidden_proportion,
            # Cloud noise
            "num_points":num_points,
            "radius":radius,
            "num_traj":num_traj,
            # Model specific
            "iterations":cfg["MRP_iterations"],
            "model":model
        }
        result_grids, runtimes = run_experiments_3D(grid,f_grid,"WP_MRP",num_runs,
                                                        params,hidden_method=hidden_method)
        result = result_grids[0,:,:,:]
        error = np.mean(np.absolute(result-grid))
        return(error)
    
    
    
    
smac = SMAC4AC(scenario=scenario, tae_runner=run_config_trajectories)

incumbent = smac.optimize()