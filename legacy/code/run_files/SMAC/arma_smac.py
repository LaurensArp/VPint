# Import libraries

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from sklearn import linear_model
from sklearn.svm import SVR

import autosklearn.regression

from importlib import reload
import resource
import os

from preprocessing.load_spatial_data import *
from preprocessing.region_graphs import *
from preprocessing.regions import *
from preprocessing.types import *
from preprocessing.label import *

from baselines.SAR import *
from baselines.MA import *
from baselines.ARMA import *

from mrp.mrp_misc import *
from mrp.mrp import *
from baselines.general import *

import smac
from smac.configspace import ConfigurationSpace
from smac.scenario.scenario import Scenario
from ConfigSpace.conditions import InCondition
from ConfigSpace.hyperparameters import CategoricalHyperparameter,UniformFloatHyperparameter, UniformIntegerHyperparameter
from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.facade.smac_ac_facade import SMAC4AC



# ----------------------------------------------------------------------------------



# Global parameters


experiment_name = "experiment_name"

# File paths



label_paths = ["/anonymised/path/labels/GDP/GDP.tif",
               "/anonymised/path/labels/covid/PatientRoute.csv"]

temp_file_path = "/anonymised/path/temp_files"

optimisation_path = "/anonymised/path/optimisation"
working_path = "/anonymised/path/working"

taxonomy_filename = "type_taxonomy_v1.tsv"



# Label parameters

label_bands = [1,None]

# GDP
#label_ind = 0
#label_from_grid = True

# COVID
label_ind = 1
label_from_grid = False

label_set = label_paths[label_ind].split("/")[-1].split(".")[0] # atrocious line I know but it's an easy solution


# Spatial parameters

region_size_lat = 0.1 # Degrees on map
region_size_lon = 0.1 # Degrees on map
location_size_lat = 0.02 # Degrees on map
location_size_lon = 0.02 # Degrees on map

region_params = [region_size_lat,region_size_lon,location_size_lat,location_size_lon]
#region_params = [location_size_lat,location_size_lon]
region_min_objects = 0

# MRP parameters

hidden_proportion = 0.8
average_loss = True



# SMAC parameters

max_runtime = 36 # hours
max_memory_per_run = 5 # GB



##################################
# Bounding boxes #################
##################################

# Taiwan bboxes

bbox_bl = (120.000,21.800)
bbox_tr = (122.058,25.389)

bbox_lower_taiwan_bl = (119.844836,21.796110)
bbox_lower_taiwan_tr = (121.965197,24.842462)

# Taipei bboxes

bbox_taipei_bl = (121.3485,24.8192)
bbox_taipei_tr = (121.7760,25.2465)

# Taichung bboxes

bbox_taichung_bl = (120.3854,23.9724)
bbox_taichung_tr = (120.9129,24.3997)

# Seoul bboxes

bbox_seoul_bl = (126.7938,37.4378)
bbox_seoul_tr = (127.3454,37.7072)

# Daegu bboxes

bbox_daegu_bl = (128.4298,35.7642)
bbox_daegu_tr = (128.7956,35.9772)

# Lower Korea bboxes

bbox_korea_lower_bl = (125.653,33.712)
bbox_korea_lower_tr = (129.813,36.129)

# Upper Korea bboxes 

bbox_korea_upper_bl = (125.938,36.053)
bbox_korea_upper_tr = (129.709,38.636)

# Amsterdam

bbox_amsterdam_bl = (4.6760,52.1914)
bbox_amsterdam_tr = (5.0919,52.5237)



# Train/test split bblox selection

bbox_train_bl = bbox_seoul_bl
bbox_train_tr = bbox_seoul_tr

bbox_test_bl = bbox_daegu_bl
bbox_test_tr = bbox_daegu_tr


# Shapefiles

shapefile_path_train = "/anonymised/path/shapefiles/Korea"
shapefile_path_test = "/anonymised/path/shapefiles/Korea"


# For logging

training_set_name = ""
if(bbox_train_bl == bbox_taipei_bl):
    training_set_name = "Taipei"
elif(bbox_train_bl == bbox_seoul_bl):
    training_set_name = "Seoul"
elif(bbox_train_bl == bbox_daegu_bl):
    training_set_name = "Daegu"
elif(bbox_train_bl == bbox_amsterdam_bl):
    training_set_name = "Amsterdam"

test_set_name = ""
if(bbox_test_bl == bbox_taipei_bl):
    test_set_name = "Taipei"
elif(bbox_test_bl == bbox_seoul_bl):
    test_set_name = "Seoul"
elif(bbox_test_bl == bbox_daegu_bl):
    test_set_name = "Daegu"
elif(bbox_test_bl == bbox_amsterdam_bl):
    test_set_name = "Amsterdam"



##################################
# Auto-sklearn parameters ########
##################################

# Just because of annoying cases where it doesn't get deleted
autoskl_current_id = 1
if(not(os.path.exists("/anonymised/path/autosklearn/id.txt"))): 
    with open("/anonymised/path/autosklearn/id.txt",'w') as fp:
        fp.write("2")
else:
    with open("/anonymised/path/autosklearn/id.txt",'r') as fp:
        s = fp.read()
        autoskl_current_id = int(s)
    with open("/anonymised/path/autosklearn/id.txt",'w') as fp:
        fp.write(str(autoskl_current_id+1))
    
    
# Weight prediction

autoskl_max_time_weights = 120
autoskl_max_time_per_run_weights = 30
autoskl_tmp_folder_weights = "/anonymised/path/autosklearn/weights/" + str(autoskl_current_id) + "temp"
autoskl_output_folder_weights = "/anonymised/path/autosklearn/weights/" + str(autoskl_current_id) + "out"

autoskl_params_weights = {}
autoskl_params_weights["max_time"] = autoskl_max_time_weights
autoskl_params_weights["max_time_per_run"] = autoskl_max_time_per_run_weights
autoskl_params_weights["tmp_folder"] = autoskl_tmp_folder_weights
autoskl_params_weights["output_folder"] = autoskl_output_folder_weights
autoskl_params_weights["delete_temp"] = True
autoskl_params_weights["delete_output"] = True

# Simple regression

autoskl_max_time_simple = 120
autoskl_max_time_per_run_simple = 30
autoskl_tmp_folder_simple = "/anonymised/path/autosklearn/simple/" + str(autoskl_current_id) + "temp"
autoskl_output_folder_simple = "/anonymised/path/autosklearn/simple/" + str(autoskl_current_id) + "out"

autoskl_params_simple = {}
autoskl_params_simple["max_time"] = autoskl_max_time_simple
autoskl_params_simple["max_time_per_run"] = autoskl_max_time_per_run_simple
autoskl_params_simple["tmp_folder"] = autoskl_tmp_folder_simple
autoskl_params_simple["output_folder"] = autoskl_output_folder_simple
autoskl_params_simple["delete_temp"] = True
autoskl_params_simple["delete_output"] = True


# Simple neighbour regression

autoskl_max_time_simple_neighbours = 120
autoskl_max_time_per_run_simple_neighbours = 30
autoskl_tmp_folder_simple_neighbours = "/anonymised/path/autosklearn/simple_neighbours/" + str(autoskl_current_id) + "temp"
autoskl_output_folder_simple_neighbours = "/anonymised/path/autosklearn/simple_neighbours/" + str(autoskl_current_id) + "out"

autoskl_params_simple_neighbours = {}
autoskl_params_simple_neighbours["max_time"] = autoskl_max_time_simple_neighbours
autoskl_params_simple_neighbours["max_time_per_run"] = autoskl_max_time_per_run_simple_neighbours
autoskl_params_simple_neighbours["tmp_folder"] = autoskl_tmp_folder_simple_neighbours
autoskl_params_simple_neighbours["output_folder"] = autoskl_output_folder_simple_neighbours
autoskl_params_simple_neighbours["delete_temp"] = True
autoskl_params_simple_neighbours["delete_output"] = True




# ----------------------------------------------------------------------------------

# SMAC search space setup

cs = ConfigurationSpace()




# Variable parameters


# Methods

missing_value_method_smac = CategoricalHyperparameter("missing_value_method", ["replace","drop"], default_value="drop")
type_filter_method_smac = CategoricalHyperparameter("type_filter_method", ["frequency","top_percent",
                                                                          "top_variable","taxonomy","none"],
                                                    default_value="top_variable")
feature_normalisation_method_smac = CategoricalHyperparameter("feature_normalisation_method", ["unit","z_score",
                                                                          "mean_norm","none"],
                                                    default_value="mean_norm")

cs.add_hyperparameters([missing_value_method_smac,type_filter_method_smac,feature_normalisation_method_smac])



# Type parameters

type_frequency_ratio_smac = UniformFloatHyperparameter("type_frequency_ratio", 0.0001, 1, default_value=0.01)
type_top_n_percent_smac = UniformIntegerHyperparameter("type_top_n_percent", 1, 100, default_value=20)
type_top_n_variable_smac = UniformIntegerHyperparameter("type_top_n_variable", 1, 50, default_value=47)

use_type_frequency_smac = InCondition(child=type_frequency_ratio_smac, parent=type_filter_method_smac,values=["frequency"])
use_type_top_n_percent_smac = InCondition(child=type_top_n_percent_smac, parent=type_filter_method_smac,values=["top_percent"])
use_type_top_n_variable_smac = InCondition(child=type_top_n_variable_smac, parent=type_filter_method_smac,values=["top_variable"])

#region_min_objects_smac = UniformIntegerHyperparameter("region_min_objects", 0, 10, default_value=1)

cs.add_hyperparameters([type_frequency_ratio_smac,type_top_n_percent_smac,type_top_n_variable_smac])
cs.add_conditions([use_type_frequency_smac,use_type_top_n_percent_smac,use_type_top_n_variable_smac])









# ----------------------------------------------------------------------------------


max_runtime_seconds = int(max_runtime * 60 * 60)
max_memory_usage_mb = max_memory_per_run * 1024


smac_current_id = 1
if(not(os.path.exists("/anonymised/path/smac_output/id.txt"))): 
    with open("/anonymised/path/smac_output/id.txt",'w') as fp:
        fp.write("2")
else:
    with open("/anonymised/path/smac_output/id.txt",'r') as fp:
        s = fp.read()
        smac_current_id = int(s)
    with open("/anonymised/path/smac_output/id.txt",'w') as fp:
        fp.write(str(smac_current_id+1))
        
if(not(os.path.exists("/anonymised/path/smac_output/conditions.csv"))): 
    with open("/anonymised/path/smac_output/conditions.csv",'w') as fp:
        fp.write("id,label_set,method,training_set_name,test_set_name\n")
with open("/anonymised/path/smac_output/conditions.csv",'a') as fp:
    s = str(smac_current_id) + "," + label_set + ",simple," + training_set_name + "," + test_set_name + "\n"
    fp.write(s)
    
        

scenario = Scenario({"run_obj": "quality",
                     "output_dir": "/anonymised/path/smac_output/" + label_set + "/ARMA/" + str(smac_current_id),
                     "wallclock_limit":max_runtime_seconds, 
                     "cs":cs
                    })




                    
# ----------------------------------------------------------------------------------
                    
def run_config_simple(cfg):
    
    # Process config
     
    cfg = {k: cfg[k] for k in cfg if cfg[k]}
    
        
    # Parameter list

    # Kinda hacky
    if(cfg["type_filter_method"] == "frequency"):
        type_params = [cfg["type_frequency_ratio"],None,None,None]
    elif(cfg["type_filter_method"] == "top_n"):
        type_params = [None,cfg["type_top_n"],None,None]
    elif(cfg["type_filter_method"] == "top_percent"):
        type_params = [None,None,cfg["type_top_n_percent"], None]
    elif(cfg["type_filter_method"] == "top_variable"):
        type_params = [None,None,None,cfg["type_top_n_variable"]]
    else:
        type_params = [None,None,None,None]

    
    

    
    # Just because of annoying cases where it doesn't get deleted
    autoskl_current_id = 1
    if(not(os.path.exists("/anonymised/path/autosklearn/id.txt"))): 
        with open("/anonymised/path/autosklearn/id.txt",'w') as fp:
            fp.write("2")
    else:
        with open("/anonymised/path/autosklearn/id.txt",'r') as fp:
            s = fp.read()
            autoskl_current_id = int(s)
        with open("/anonymised/path/autosklearn/id.txt",'w') as fp:
            fp.write(str(autoskl_current_id+1))
    
    autoskl_max_time = 120
    autoskl_max_time_per_run = 30
    autoskl_tmp_folder = "/anonymised/path/autosklearn/weights/spatial/" + str(autoskl_current_id) + "temp"
    autoskl_output_folder = "/anonymised/path/autosklearn/weights/spatial/" + str(autoskl_current_id) + "out"

    autoskl_params_sar = {}
    autoskl_params_sar["max_time"] = autoskl_max_time
    autoskl_params_sar["max_time_per_run"] = autoskl_max_time_per_run
    autoskl_params_sar["tmp_folder"] = autoskl_tmp_folder
    autoskl_params_sar["output_folder"] = autoskl_output_folder
    autoskl_params_sar["delete_temp"] = True
    autoskl_params_sar["delete_output"] = True


    # Just because of annoying cases where it doesn't get deleted
    autoskl_current_id = 1
    if(not(os.path.exists("/anonymised/path/autosklearn/id.txt"))): 
        with open("/anonymised/path/autosklearn/id.txt",'w') as fp:
            fp.write("2")
    else:
        with open("/anonymised/path/autosklearn/id.txt",'r') as fp:
            s = fp.read()
            autoskl_current_id = int(s)
        with open("/anonymised/path/autosklearn/id.txt",'w') as fp:
            fp.write(str(autoskl_current_id+1))



    autoskl_max_time = 120
    autoskl_max_time_per_run = 30
    autoskl_tmp_folder = "/anonymised/path/autosklearn/weights/spatial/" + str(autoskl_current_id) + "temp"
    autoskl_output_folder = "/anonymised/path/autosklearn/weights/spatial/" + str(autoskl_current_id) + "out"

    autoskl_params_arma = {}
    autoskl_params_arma["max_time"] = autoskl_max_time
    autoskl_params_arma["max_time_per_run"] = autoskl_max_time_per_run
    autoskl_params_arma["tmp_folder"] = autoskl_tmp_folder
    autoskl_params_arma["output_folder"] = autoskl_output_folder
    autoskl_params_arma["delete_temp"] = True
    autoskl_params_arma["delete_output"] = True
    
    
    # Preprocessing training set

    S = load_spatial_data(shapefile_path_train,cfg["missing_value_method"])
    S = clip_area(S,bbox_train_bl,bbox_train_tr)

    S,types = find_types(S,optimisation_path,working_path,cfg["type_filter_method"],type_params,
                         taxonomy_filename=taxonomy_filename,verbose=False)

    S = compute_centroids(S)
    region_bounds = compute_region_bounds(S,location_size_lat,location_size_lon)
    regions,region_bounds = assign_objects_to_regions(S,region_bounds,region_min_objects=region_min_objects)
    super_G = create_super_graph_raw(regions,region_bounds,types,location_size_lat,location_size_lon)

    super_H_train,width_train,height_train = convert_super_G(super_G,S,label_paths[label_ind],region_params,hidden_proportion,
                                                            from_grid=label_from_grid)


    # Preprocessing test set

    S = load_spatial_data(shapefile_path_test,cfg["missing_value_method"])
    S = clip_area(S,bbox_test_bl,bbox_test_tr)

    S,types = find_types(S,optimisation_path,working_path,cfg["type_filter_method"],type_params,
                         taxonomy_filename=taxonomy_filename,verbose=False)

    S = compute_centroids(S)
    region_bounds = compute_region_bounds(S,location_size_lat,location_size_lon)
    regions,region_bounds = assign_objects_to_regions(S,region_bounds,region_min_objects=region_min_objects)
    super_G = create_super_graph_raw(regions,region_bounds,types,location_size_lat,location_size_lon)

    super_H_test,width_test,height_test = convert_super_G(super_G,S,label_paths[label_ind],region_params,hidden_proportion,
                                                            from_grid=label_from_grid)

    
    # ARMA code
    
    W_train, y_mean_train = weight_matrix_rook(super_H_train)
    m, automl = train_ARMA(super_H_train,W_train,y_mean_train,autoskl_params_sar,autoskl_params_arma)
    
    W_test, y_mean_test = weight_matrix_rook(super_H_test)
    error = test_ARMA(super_H_test,W_test,y_mean_test,m,automl)

    return(error)

    
    
    
# ----------------------------------------------------------------------------------
    
# Run SMAC

smac = SMAC4AC(scenario=scenario, tae_runner=run_config_simple)

incumbent = smac.optimize()
