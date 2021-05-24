# Import libraries

from preprocessing.load_spatial_data import *
from preprocessing.region_graphs import *
from preprocessing.regions import *
from preprocessing.types import *
from preprocessing.label import *

from mrp.mrp_misc import *
from mrp.mrp import *
from baselines.general import *
from baselines.CNN import *

import pandas as pd
import numpy as np
import networkx as nx

from sklearn.metrics import mean_absolute_error as mae
import autokeras as ak

import os

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

label_bands = [1,1,1,1,1,None]

# GDP
label_ind = 0
label_from_grid = True

# COVID
#label_ind = 1
#label_from_grid = False

label_set = label_paths[label_ind].split("/")[-1].split(".")[0] # atrocious line I know but it's an easy solution


# Spatial parameters

region_size_lat = 0.1 # Degrees on map
region_size_lon = 0.1 # Degrees on map
location_size_lat = 0.02 # Degrees on map
location_size_lon = 0.02 # Degrees on map

region_params = [region_size_lat,region_size_lon,location_size_lat,location_size_lon]
#region_params = [location_size_lat,location_size_lon]





hidden_proportion = 0.8



# SMAC parameters

max_runtime = 36 # hours
max_memory_per_run = 5 # GB


##################################
# Bounding boxes #################
##################################

# Taipei bboxes

bbox_taipei_bl = (121.3485,24.8192)
bbox_taipei_tr = (121.7760,25.2465)
shp = "/anonymised/path/shapefiles/Taiwan"
taipei_dict = {"bbox_bl":bbox_taipei_bl,"bbox_tr":bbox_taipei_tr,"shp":shp}

# Taichung bboxes

bbox_taichung_bl = (120.3854,23.9724)
bbox_taichung_tr = (120.9129,24.3997)
shp = "/anonymised/path/shapefiles/Taiwan"
taichung_dict = {"bbox_bl":bbox_taichung_bl,"bbox_tr":bbox_taichung_tr,"shp":shp}

# Seoul bboxes

bbox_seoul_bl = (126.7938,37.4378)
bbox_seoul_tr = (127.3454,37.7072)
shp = "/anonymised/path/shapefiles/Korea"
seoul_dict = {"bbox_bl":bbox_seoul_bl,"bbox_tr":bbox_seoul_tr,"shp":shp}

# Daegu bboxes

bbox_daegu_bl = (128.4298,35.7642)
bbox_daegu_tr = (128.7956,35.9772)
shp = "/anonymised/path/shapefiles/Korea"
daegu_dict = {"bbox_bl":bbox_daegu_bl,"bbox_tr":bbox_daegu_tr,"shp":shp}

# Busan bboxes

bbox_busan_bl = (128.7678,35.0110)
bbox_busan_tr = (129.4241,35.4155)
shp = "/anonymised/path/shapefiles/Korea"
busan_dict = {"bbox_bl":bbox_busan_bl,"bbox_tr":bbox_busan_tr,"shp":shp}

# Amsterdam

bbox_amsterdam_bl = (4.6760,52.1914)
bbox_amsterdam_tr = (5.0919,52.5237)
shp = "/anonymised/path/shapefiles/Amsterdam"
amsterdam_dict = {"bbox_bl":bbox_amsterdam_bl,"bbox_tr":bbox_amsterdam_tr,"shp":shp}


regions = [taipei_dict,taichung_dict,seoul_dict,daegu_dict,busan_dict,amsterdam_dict]

shapefile_path_train = seoul_dict['shp']
bbox_train_bl = seoul_dict['bbox_bl']
bbox_train_tr = seoul_dict['bbox_tr']

shapefile_path_test = daegu_dict['shp']
bbox_test_bl = daegu_dict['bbox_bl']
bbox_test_tr = daegu_dict['bbox_tr']



training_set_name = ""
if(bbox_train_bl == bbox_taipei_bl):
    training_set_name = "Taipei"
elif(bbox_train_bl == bbox_taichung_bl):
    training_set_name = "Taichung"
elif(bbox_train_bl == bbox_seoul_bl):
    training_set_name = "Seoul"
elif(bbox_train_bl == bbox_daegu_bl):
    training_set_name = "Daegu"
elif(bbox_train_bl == bbox_busan_bl):
    training_set_name = "Busan"
elif(bbox_train_bl == bbox_amsterdam_bl):
    training_set_name = "Amsterdam"

test_set_name = ""
if(bbox_test_bl == bbox_taipei_bl):
    test_set_name = "Taipei"
elif(bbox_test_bl == bbox_taichung_bl):
    test_set_name = "Taichung"
elif(bbox_test_bl == bbox_seoul_bl):
    test_set_name = "Seoul"
elif(bbox_test_bl == bbox_daegu_bl):
    test_set_name = "Daegu"
elif(bbox_test_bl == bbox_busan_bl):
    test_set_name = "Busan"
elif(bbox_test_bl == bbox_amsterdam_bl):
    test_set_name = "Amsterdam"
  








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


cs.add_hyperparameters([type_frequency_ratio_smac,type_top_n_percent_smac,type_top_n_variable_smac])
cs.add_conditions([use_type_frequency_smac,use_type_top_n_percent_smac,use_type_top_n_variable_smac])


nn_window_height_smac = UniformIntegerHyperparameter("nn_window_height", 1, 20, default_value=5)
nn_window_width_smac = UniformIntegerHyperparameter("nn_window_width", 1, 20, default_value=5)
nn_validation_split_smac = UniformFloatHyperparameter("nn_validation_split", 0, 1, default_value=0.2)

cs.add_hyperparameters([nn_window_height_smac,nn_window_width_smac,nn_validation_split_smac])






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
    s = str(smac_current_id) + "," + label_set + ",CNN," + training_set_name + "," + test_set_name + "\n"
    fp.write(s)
    
        

scenario = Scenario({"run_obj": "quality",
                     "output_dir": "/anonymised/path/smac_output/" + label_set + "/CNN/" + str(smac_current_id),
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

    hidden_proportion_test = hidden_proportion
    hidden_proportion_train = hidden_proportion
    
    # Preprocessing training set

    S = load_spatial_data(shapefile_path_train,cfg["missing_value_method"])
    S = clip_area(S,bbox_train_bl,bbox_train_tr)

    S,types = find_types(S,optimisation_path,working_path,cfg["type_filter_method"],type_params,
                         taxonomy_filename=taxonomy_filename,verbose=False)

    S = compute_centroids(S)
    region_bounds = compute_region_bounds(S,location_size_lat,location_size_lon)
    regions,region_bounds = assign_objects_to_regions(S,region_bounds,region_min_objects=0)
    super_G = create_super_graph_raw(regions,region_bounds,types,location_size_lat,location_size_lon)
    super_H_train,width_train,height_train = convert_super_G(super_G,S,label_paths[label_ind],region_params,hidden_proportion_train,
                                                                        from_grid=label_from_grid)



    S = load_spatial_data(shapefile_path_test,cfg["missing_value_method"])
    S = clip_area(S,bbox_test_bl,bbox_test_tr)

    S,types = find_types(S,optimisation_path,working_path,cfg["type_filter_method"],type_params,
                         taxonomy_filename=taxonomy_filename,verbose=False)

    S = compute_centroids(S)
    region_bounds = compute_region_bounds(S,location_size_lat,location_size_lon)
    regions,region_bounds = assign_objects_to_regions(S,region_bounds,region_min_objects=0)
    super_G = create_super_graph_raw(regions,region_bounds,types,location_size_lat,location_size_lon)
    super_H_test,width_test,height_test = convert_super_G(super_G,S,label_paths[label_ind],region_params,hidden_proportion_test,
                                                                        from_grid=label_from_grid)

    
    # NN code
    
    super_H_train = shuffle_hidden(super_H_train,hidden_proportion_train)
    super_H_test = shuffle_hidden(super_H_test,hidden_proportion)

    X_train, y_train = graph_to_tensor_train(super_H_train,cfg["nn_window_height"],cfg["nn_window_width"])
    X_test, y_test = graph_to_tensor_test(super_H_test,cfg["nn_window_height"],cfg["nn_window_width"])
    
    reg = ak.ImageRegressor(
        overwrite=True,
        directory="/anonymised/path/autokeras/",
        loss="mean_absolute_error")

    reg.fit(X_train, y_train, validation_split=cfg["nn_validation_split"])
    
    pred = reg.predict(X_test)
    error = mae(pred,y_test)

    return(error)

    
    
    
# ----------------------------------------------------------------------------------
    
# Run SMAC

smac = SMAC4AC(scenario=scenario, tae_runner=run_config_simple)

incumbent = smac.optimize()
