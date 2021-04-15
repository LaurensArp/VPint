from utils.generate_data import *
from utils.hide_spatial_data import *
from utils.experiments import run_experiments_2D, run_experiments_3D
from utils.load_shapefile_features import *
from utils.load_trajectories import *
from utils.load_msi import *

import smac
from smac.configspace import ConfigurationSpace
from smac.scenario.scenario import Scenario
from ConfigSpace.conditions import InCondition
from ConfigSpace.hyperparameters import CategoricalHyperparameter,UniformFloatHyperparameter, UniformIntegerHyperparameter
from smac.facade.smac_ac_facade import SMAC4AC

from sklearn.linear_model import LinearRegression
import autosklearn.regression
import autokeras as ak

import rasterio
import sys

from MRPinterpolation.SD_MRP import SD_SMRP, SD_STMRP
from MRPinterpolation.WP_MRP import WP_SMRP, WP_STMRP




########################################
# PARAMETERS
########################################




# Setup

# General

#setting_name = "test_run"
setting_name = sys.argv[1]

# Valid: "porto_trajectories", "synthetic_2D", "synthetic_3D", "GDP", "covid", "satellites"
dataset = sys.argv[2]

# Valid: "random", "clouds"
hidden_method = sys.argv[4]

# Valid: "OK", "UK", "basic", "SAR", "MA", "ARMA", "CNN", "SD_MRP", "WP_MRP"
alg = sys.argv[3]

hidden_proportion = 0.8

num_points = 5
radius =  4
num_traj = 10

taxonomy_path = "/mnt/e/User Files/Projects/Master Thesis/Working/type_taxonomy_v1.tsv"


# SMAC

max_runtime = int(sys.argv[5]) # hours
max_memory_per_run = 5 # GB



# Trajectories

traj_path_original = "/mnt/c/Users/Laurens/Projects/Porto Trajectories/porto_trajectories_all.csv"
traj_path = "/mnt/c/Users/Laurens/Projects/Porto Trajectories/porto_trajectories_preprocessed.csv"

traj_spatial_path = "/mnt/c/Users/Laurens/Projects/Portugal Shapefiles/"
traj_spatial_sources = ["gis_osm_buildings_a_free_1.shp",
                   "gis_osm_natural_free_1.shp",
                  "gis_osm_places_free_1.shp",
                  "gis_osm_pofw_free_1.shp",
                  "gis_osm_traffic_a_free_1.shp",
                  "gis_osm_transport_a_free_1.shp"]

traj_spatial_data = [(traj_spatial_path + s) for s in traj_spatial_sources]

traj_bbox_tl = (41.249814,-8.734482)
traj_bbox_br = (41.056908,-8.502107)

traj_res_y = 100
traj_res_x = 100


# Synthetic 2D

synth2d_params = {
    "param_grid_height":20,
    "param_grid_width":20,
    "param_feature_correlation":0.5,
    "param_num_features":2
}



# Synthetic 3D

synth3d_params = {
    "param_grid_height":100,
    "param_grid_width":100,
    "param_grid_depth":10,
    "param_temporal_autocorr":0.25,
    "param_feature_correlation":0.5,
    "param_num_features":2
}

# GDP


# Covid


# Satellites

sat_most_recent_path = "/mnt/c/Users/Laurens/Projects/Satellite Data/Mediterranean/Crops/GeoTIFF/most_recent.tif"
sat_week_1_path = "/mnt/c/Users/Laurens/Projects/Satellite Data/Mediterranean/Crops/GeoTIFF/1_week.tif"
sat_month_1_path = "/mnt/c/Users/Laurens/Projects/Satellite Data/Mediterranean/Crops/GeoTIFF/1_month.tif"
sat_month_6_path = "/mnt/c/Users/Laurens/Projects/Satellite Data/Mediterranean/Crops/GeoTIFF/6_months.tif"
sat_month_12_path = "/mnt/c/Users/Laurens/Projects/Satellite Data/Mediterranean/Crops/GeoTIFF/12_months.tif"

paths = [sat_most_recent_path,sat_week_1_path,sat_month_1_path,sat_month_6_path,sat_month_12_path]





########################################
# CONFIG SPACE
########################################



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
    if(dataset == "GDP" or dataset == "covid" or dataset == "porto_trajectories"):
        cs.add_hyperparameters([missing_value_method_smac,type_filter_method_smac,feature_normalisation_method_smac])



# Type parameters

type_top_frequent_smac = UniformIntegerHyperparameter("type_top_frequent", 1, 100, default_value=10)

use_type_top_frequent_smac = InCondition(child=type_top_frequent_smac, parent=type_filter_method_smac,values=["top_frequent"])

if(alg != "OK" and  alg != "UK" and alg != "SD_MRP"):
    if(dataset == "GDP" or dataset == "covid" or dataset == "porto_trajectories"):
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

elif(alg == "CNN"):
    MRP_iterations_smac = UniformIntegerHyperparameter("MRP_iterations", 1, 200, default_value=50)
    cs.add_hyperparameters([MRP_iterations_smac])
    
    nn_max_trials_smac = UniformIntegerHyperparameter("nn_max_trials", 1, 200, default_value=5)
    nn_epochs_smac = UniformIntegerHyperparameter("nn_epochs", 1, 200, default_value=5)
    nn_train_fill_smac = CategoricalHyperparameter("nn_train_fill",["True","False"],default_value="True")
    nn_window_height_smac = UniformIntegerHyperparameter("nn_window_height", 1, 50, default_value=5)
    nn_window_width_smac = UniformIntegerHyperparameter("nn_window_width", 1, 50, default_value=5)
    cs.add_hyperparameters([nn_max_trials_smac,nn_epochs_smac,nn_train_fill_smac,nn_window_height_smac,nn_window_width_smac])

    
    
    
    
    
########################################
# SMAC SETUP
########################################

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
    
autosklearn_store_path = "smac/smac_output/" + setting_name + "/" + dataset + "/" + alg + "/" + str(smac_current_id) + "/autosklearn/"
autokeras_store_path = "smac/smac_output/" + setting_name + "/" + dataset + "/" + alg + "/" + str(smac_current_id) + "/autokeras/"

scenario = Scenario({"run_obj": "quality",
                     "output_dir": "smac/smac_output/" + setting_name + "/" + dataset + "/" + alg + "/" + str(smac_current_id),
                     "wallclock_limit":max_runtime_seconds, 
                     "cs":cs
                    })



########################################
# RUN FUNCTIONS
########################################


def run_general(grid,f_grid,cfg):
    if(alg != "OK" and  alg != "UK" and alg != "SD_MRP"):
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
            tmp_folder=autosklearn_store_path + str(autoskl_current_id) + "temp",
            output_folder=autosklearn_store_path + str(autoskl_current_id) + "out",
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
        f_grid = None
    
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
        f_grid = None
    
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
        f_grid = None
        
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
    
    elif(alg == "CNN"):
        reg = ak.ImageRegressor(
            overwrite=True, 
            max_trials=cfg["nn_max_trials"],
            directory=autokeras_store_path,
            project_name=setting_name,
        )
        params = {
            # Random noise
            "hidden_proportion":hidden_proportion,
            # Cloud noise
            "num_points":num_points,
            "radius":radius,
            "num_traj":num_traj,
            # Model specific
            "nn_model":reg,
            "nn_epochs":cfg["nn_epochs"], 
            "nn_train_fill":cfg["nn_train_fill"],
            "nn_window_height":cfg["nn_window_height"],
            "nn_window_width":cfg["nn_window_width"],
        }    
    
    if(dataset == "porto_trajectories"):
        result_grids, runtimes = run_experiments_3D(grid,f_grid,alg,1,
                                                        params,hidden_method=hidden_method)
        result = result_grids[0,:,:,:]
    else:
        result_grids, runtimes = run_experiments_2D(grid,f_grid,alg,1,
                                                        params,hidden_method=hidden_method)
        result = result_grids[0,:,:]
    error = np.mean(np.absolute(result-grid))
    return(error)
    
    
def run_config_trajectories(cfg):
    cfg = {k: cfg[k] for k in cfg if cfg[k]}

    if(alg != "OK" and  alg != "UK" and alg != "SD_MRP"):
        additional_params = {}
        if(cfg["type_filter_method"] == "taxonomy"):
            additional_params["taxonomy_path"] = taxonomy_path
        elif(cfg["type_filter_method"] == "top_frequent"):
            additional_params["num_features"] = cfg["type_top_frequent"]

    # Preprocess
    
    df = pd.read_csv(traj_path)
    df = filter_bbox(df,traj_bbox_tl,traj_bbox_br)
    meta = get_meta(traj_res_y,traj_res_x,traj_bbox_tl,traj_bbox_br)
    grid = assign_traj_to_grid(df,meta)
    f_grid = None

    if(alg != "OK" and  alg != "UK" and alg != "SD_MRP"):
        meta = get_meta(traj_res_y,traj_res_x,traj_bbox_tl,traj_bbox_br)
        S = load_spatial_data(traj_spatial_data,traj_bbox_tl,traj_bbox_br,cfg["missing_value_method"])
        f_grid = assign_shapes_to_f_grid(S,meta,cfg["type_filter_method"],additional_params=additional_params)
        f_grid = normalise_attributes(f_grid,cfg["feature_normalisation_method"])
        
    error = run_general(grid,f_grid,cfg)
    return(error)


def run_config_synthetic_2D(cfg):
    cfg = {k: cfg[k] for k in cfg if cfg[k]}

    # Preprocess
    
    grid, f_grid = generate_data(user_params=synth2d_params,generate_features=True)
    
    error = run_general(grid,f_grid,cfg)
    return(error)


def run_config_synthetic_3D(cfg):
    cfg = {k: cfg[k] for k in cfg if cfg[k]}

    # Preprocess
    
    grid, f_grid = generate_3D_data(user_params=synth3d_params,generate_features=True)
    error = run_general(grid,f_grid,cfg)
    return(error)



def run_config_GDP(cfg):
    cfg = {k: cfg[k] for k in cfg if cfg[k]}
    
    if(alg != "OK" and  alg != "UK" and alg != "SD_MRP"):
        additional_params = {}
        if(cfg["type_filter_method"] == "taxonomy"):
            additional_params["taxonomy_path"] = taxonomy_path
        elif(cfg["type_filter_method"] == "top_frequent"):
            additional_params["num_features"] = cfg["type_top_frequent"]

    # Preprocess
    
    # TODO
    
    error = run_general(grid,f_grid,cfg)
    return(error)


def run_config_covid(cfg):
    cfg = {k: cfg[k] for k in cfg if cfg[k]}
    
    if(alg != "OK" and  alg != "UK" and alg != "SD_MRP"):
        additional_params = {}
        if(cfg["type_filter_method"] == "taxonomy"):
            additional_params["taxonomy_path"] = taxonomy_path
        elif(cfg["type_filter_method"] == "top_frequent"):
            additional_params["num_features"] = cfg["type_top_frequent"]

    # Preprocess
    
    # TODO
    
    error = run_general(grid,f_grid,cfg)
    return(error)



def run_config_satellites(cfg):
    cfg = {k: cfg[k] for k in cfg if cfg[k]}

    # Preprocess
    
    data = load_satellite_data(paths,normalise=True)
    grid, f_grid = msi_to_grid(data,target_index=0,band_index=0,generate_features=True)
    
    error = run_general(grid,f_grid,cfg)
    return(error)



########################################
# CALL SMAC
########################################


if(dataset == "porto_trajectories"):
    smac = SMAC4AC(scenario=scenario, tae_runner=run_config_trajectories)
elif(dataset == "covid"):
    smac = SMAC4AC(scenario=scenario, tae_runner=run_config_covid)
elif(dataset == "GDP"):
    smac = SMAC4AC(scenario=scenario, tae_runner=run_config_GDP)
elif(dataset == "satellites"):
    smac = SMAC4AC(scenario=scenario, tae_runner=run_config_satellites)
elif(dataset == "synthetic_2D"):
    smac = SMAC4AC(scenario=scenario, tae_runner=run_config_synthetic_2D)
elif(dataset == "synthetic_3D"):
    smac = SMAC4AC(scenario=scenario, tae_runner=run_config_synthetic_3D)

incumbent = smac.optimize()