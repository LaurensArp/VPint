# Import libraries

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from sklearn import linear_model
from sklearn.svm import SVR

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
from baselines.CNN import *

from sklearn.metrics import mean_absolute_error as mae
import autokeras as ak
from tensorflow.keras.models import load_model
import tensorflow as tf






# Global parameters





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



# Optimisation parameters

MRP_iter = 100
optimisation_epochs = 100
average_loss = True
mutation_rate = 0.2
mutation_intensity = 0.5
train_proportion = 0.8


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



# Train/test split bblox selection

bbox_train_bl = bbox_taichung_bl
bbox_train_tr = bbox_taichung_tr

bbox_test_bl = bbox_daegu_bl
bbox_test_tr = bbox_daegu_tr


# Shapefiles

shapefile_path_train = "/anonymised/path/shapefiles/Taiwan"
shapefile_path_test = "/anonymised/path/shapefiles/Korea"


# For logging

training_set_name = ""
if(bbox_train_bl == bbox_taipei_bl):
    training_set_name = "Taipei"
elif(bbox_train_bl == bbox_taichung_bl):
    training_set_name = "Taichung"
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
elif(bbox_test_bl == bbox_taichung_bl):
    test_set_name = "Taichung"
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











# Variable parameters


# Type parameters

type_frequency_ratio = 0.01
type_top_n = 10
type_top_n_percent = 20
type_top_n_variable = 15

type_params = [type_frequency_ratio,type_top_n,type_top_n_percent,type_top_n_variable]

region_min_objects = 0



# Methods

# Possible values: "replace","drop"
missing_value_method = "replace"

# Possible values: "frequency,top,top_percent,top_variable,taxonomy,none"
type_filter_method = "taxonomy"

# Possible values: "unit","z_score","mean_norm","none"
feature_normalisation_method = "unit"







# Test

#num_runs = 1
#ps = [0.5]
#cities = [seoul_dict]


# Full

num_runs = 5
ps = [0.1,0.3,0.5,0.7,0.9]
cities = [daegu_dict,taipei_dict]

hidden_proportion = 0.8 


experiment_name = "self_supervised_2"

for city in cities:

    bbox_test_bl = city['bbox_bl']
    bbox_test_tr = city['bbox_tr']
    shapefile_path_test = city['shp']

    test_set_name = ""
    if(bbox_test_bl == bbox_taipei_bl):
        test_set_name = "Taipei"
    elif(bbox_test_bl == bbox_seoul_bl):
        test_set_name = "Seoul"
    elif(bbox_test_bl == bbox_taichung_bl):
        test_set_name = "Taichung"
    elif(bbox_test_bl == bbox_daegu_bl):
        test_set_name = "Daegu"
    elif(bbox_test_bl == bbox_amsterdam_bl):
        test_set_name = "Amsterdam"
    


                    

    ############################################
    # MA
    ############################################

    # Setup result file
    if(not(os.path.exists("/anonymised/path/results/" + experiment_name))):
        os.mkdir("/anonymised/path/results/" + experiment_name)
    if(not(os.path.exists("/anonymised/path/results/" + experiment_name + "/" + label_set))):
        os.mkdir("/anonymised/path/results/" + experiment_name + "/" + label_set)
    if(not(os.path.exists("/anonymised/path/results/" + experiment_name + "/" + label_set + "/MA_" + training_set_name + "_" + 
                          test_set_name + ".csv"))): 
        with open("/anonymised/path/results/" + experiment_name + "/" + label_set + "/MA_" + training_set_name + "_" + 
                          test_set_name + ".csv",'w') as fp:
            fp.write("proportion_train,proportion_test,value\n")


    # Preprocessing variables

    feature_normalisation_method = "mean_norm"
    missing_value_method = "drop"
    type_filter_method = "top_variable"
    type_top_n_variable = 47

    # Preprocess test set

    S = load_spatial_data(shapefile_path_test,missing_value_method)
    S = clip_area(S,bbox_test_bl,bbox_test_tr)

    S,types = find_types(S,optimisation_path,working_path,type_filter_method,type_params,
                         taxonomy_filename=taxonomy_filename,verbose=False)

    S = compute_centroids(S)
    region_bounds = compute_region_bounds(S,location_size_lat,location_size_lon)
    regions,region_bounds = assign_objects_to_regions(S,region_bounds,region_min_objects=region_min_objects)
    super_G = create_super_graph_raw(regions,region_bounds,types,location_size_lat,location_size_lon)

    super_H_test,width_test,height_test = convert_super_G(super_G,S,label_paths[label_ind],region_params,hidden_proportion,
                                                            from_grid=label_from_grid)



    for prop in ps:
                                                                               
        for i in range(0,num_runs):
        
            super_H_test = shuffle_hidden(super_H_test,prop)
            
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

            autoskl_params = {}
            autoskl_params["max_time"] = autoskl_max_time
            autoskl_params["max_time_per_run"] = autoskl_max_time_per_run
            autoskl_params["tmp_folder"] = autoskl_tmp_folder
            autoskl_params["output_folder"] = autoskl_output_folder
            autoskl_params["delete_temp"] = True
            autoskl_params["delete_output"] = True
            
            W_train, y_mean_train = weight_matrix_rook(super_H_test)
            m, automl = train_MA(super_H_test,W_train,autoskl_params)
            
            flip_hidden(super_H_test)
            W_test, y_mean_test = weight_matrix_rook(super_H_test)
            error = test_MA(super_H_test,W_test,m,automl)
            
            with open("/anonymised/path/results/" + experiment_name + "/" + label_set + "/MA_" + training_set_name + "_" + 
                              test_set_name + ".csv",'a') as fp:
                s = str(None) + "," + str(prop) + "," + str(error) + "\n"
                fp.write(s)

    ############################################
    # ARMA
    ############################################

    # Setup result file
    if(not(os.path.exists("/anonymised/path/results/" + experiment_name))):
        os.mkdir("/anonymised/path/results/" + experiment_name)
    if(not(os.path.exists("/anonymised/path/results/" + experiment_name + "/" + label_set))):
        os.mkdir("/anonymised/path/results/" + experiment_name + "/" + label_set)
    if(not(os.path.exists("/anonymised/path/results/" + experiment_name + "/" + label_set + "/ARMA_" + training_set_name + "_" + 
                          test_set_name + ".csv"))): 
        with open("/anonymised/path/results/" + experiment_name + "/" + label_set + "/ARMA_" + training_set_name + "_" + 
                          test_set_name + ".csv",'w') as fp:
            fp.write("proportion_train,proportion_test,value\n")


    # Preprocessing variables

    feature_normalisation_method = "unit"
    missing_value_method = "drop"
    type_filter_method = "top_variable"
    type_top_n_variable = 48

    # Preprocess test set

    S = load_spatial_data(shapefile_path_test,missing_value_method)
    S = clip_area(S,bbox_test_bl,bbox_test_tr)

    S,types = find_types(S,optimisation_path,working_path,type_filter_method,type_params,
                         taxonomy_filename=taxonomy_filename,verbose=False)

    S = compute_centroids(S)
    region_bounds = compute_region_bounds(S,location_size_lat,location_size_lon)
    regions,region_bounds = assign_objects_to_regions(S,region_bounds,region_min_objects=region_min_objects)
    super_G = create_super_graph_raw(regions,region_bounds,types,location_size_lat,location_size_lon)

    super_H_test,width_test,height_test = convert_super_G(super_G,S,label_paths[label_ind],region_params,hidden_proportion,
                                                            from_grid=label_from_grid)



    for prop in ps:
                                                                               
        for i in range(0,num_runs):
        
            super_H_test = shuffle_hidden(super_H_test,prop)
            
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
            
            
            
            W_train, y_mean_train = weight_matrix_rook(super_H_test)
            m, automl = train_ARMA(super_H_test,W_train,y_mean_train,autoskl_params_sar,autoskl_params_arma)
            
            flip_hidden(super_H_test)
            W_test, y_mean_test = weight_matrix_rook(super_H_test)
            error = test_ARMA(super_H_test,W_test,y_mean_test,m,automl)
            
            with open("/anonymised/path/results/" + experiment_name + "/" + label_set + "/ARMA_" + training_set_name + "_" + 
                              test_set_name + ".csv",'a') as fp:
                s = str(None) + "," + str(prop) + "," + str(error) + "\n"
                fp.write(s)



    ############################################
    # CNN
    ############################################

    # Setup result file
    if(not(os.path.exists("/anonymised/path/results/" + experiment_name))):
        os.mkdir("/anonymised/path/results/" + experiment_name)
    if(not(os.path.exists("/anonymised/path/results/" + experiment_name + "/" + label_set))):
        os.mkdir("/anonymised/path/results/" + experiment_name + "/" + label_set)
    if(not(os.path.exists("/anonymised/path/results/" + experiment_name + "/" + label_set + "/CNN_" + training_set_name + "_" + 
                          test_set_name + ".csv"))): 
        with open("/anonymised/path/results/" + experiment_name + "/" + label_set + "/CNN_" + training_set_name + "_" + 
                          test_set_name + ".csv",'w') as fp:
            fp.write("proportion_train,proportion_test,value\n")


    # Variables

    missing_value_method = "replace"
    type_filter_method = "frequency"
    feature_normalisation_method = "none"
    type_frequency_ratio = 0.5887024334058893

    nn_window_height = 17
    nn_window_width = 18
    nn_validation_split = 0.33467051772273004

    # Preprocess test set

    S = load_spatial_data(shapefile_path_test,missing_value_method)
    S = clip_area(S,bbox_test_bl,bbox_test_tr)

    S,types = find_types(S,optimisation_path,working_path,type_filter_method,type_params,
                         taxonomy_filename=taxonomy_filename,verbose=False)

    S = compute_centroids(S)
    region_bounds = compute_region_bounds(S,location_size_lat,location_size_lon)
    regions,region_bounds = assign_objects_to_regions(S,region_bounds,region_min_objects=region_min_objects)
    super_G = create_super_graph_raw(regions,region_bounds,types,location_size_lat,location_size_lon)

    super_H_test,width_test,height_test = convert_super_G(super_G,S,label_paths[label_ind],region_params,hidden_proportion,
                                                            from_grid=label_from_grid)



    for prop in ps:
                                                                               
        for i in range(0,num_runs):
        
            super_H_test = shuffle_hidden(super_H_test,prop)
            
            X_train, y_train = graph_to_tensor_train(super_H_test,nn_window_height,nn_window_width)
            
            if(city == "Taipei"):
                model_original = load_model("/anonymised/path/autokeras/models/trials_50", 
                                  custom_objects=ak.CUSTOM_OBJECTS)
            else:
                model_original = load_model("/anonymised/path/autokeras/models/gdp_taichung_daegu_50", 
                                  custom_objects=ak.CUSTOM_OBJECTS)
                                  
            model = tf.keras.models.clone_model(model_original)
            model.compile(loss="mean_absolute_error")
            
            model.fit(X_train, y_train, validation_split=nn_validation_split)
            
            flip_hidden(super_H_test)
            X_test, y_test = graph_to_tensor_test(super_H_test,nn_window_height,nn_window_width)
            pred = model.predict(X_test)
            error = mae(pred,y_test)
                    

                    
            with open("/anonymised/path/results/" + experiment_name + "/" + label_set + "/CNN_" + training_set_name + "_" + 
                              test_set_name + ".csv",'a') as fp:
                s = str(None) + "," + str(prop) + "," + str(error) + "\n"
                fp.write(s)


    ############################################
    # SAR
    ############################################
    

    # Setup result file
    if(not(os.path.exists("/anonymised/path/results/" + experiment_name))):
        os.mkdir("/anonymised/path/results/" + experiment_name)
    if(not(os.path.exists("/anonymised/path/results/" + experiment_name + "/" + label_set))):
        os.mkdir("/anonymised/path/results/" + experiment_name + "/" + label_set)
    if(not(os.path.exists("/anonymised/path/results/" + experiment_name + "/" + label_set + "/SAR_" + training_set_name + "_" + 
                          test_set_name + ".csv"))): 
        with open("/anonymised/path/results/" + experiment_name + "/" + label_set + "/SAR_" + training_set_name + "_" + 
                          test_set_name + ".csv",'w') as fp:
            fp.write("proportion_train,proportion_test,value\n")


    # Preprocessing variables
    
    feature_normalisation_method = "mean_norm"
    missing_value_method = "replace"
    type_filter_method = "frequency"
    type_frequency_ratio = 0.21134122318651932

    # Preprocess test set

    S = load_spatial_data(shapefile_path_test,missing_value_method)
    S = clip_area(S,bbox_test_bl,bbox_test_tr)

    S,types = find_types(S,optimisation_path,working_path,type_filter_method,type_params,
                         taxonomy_filename=taxonomy_filename,verbose=False)

    S = compute_centroids(S)
    region_bounds = compute_region_bounds(S,location_size_lat,location_size_lon)
    regions,region_bounds = assign_objects_to_regions(S,region_bounds,region_min_objects=region_min_objects)
    super_G = create_super_graph_raw(regions,region_bounds,types,location_size_lat,location_size_lon)

    super_H_test,width_test,height_test = convert_super_G(super_G,S,label_paths[label_ind],region_params,hidden_proportion,
                                                            from_grid=label_from_grid)



    for prop in ps:
                                                                               
        for i in range(0,num_runs):
        
            super_H_test = shuffle_hidden(super_H_test,prop)
            
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

            autoskl_params = {}
            autoskl_params["max_time"] = autoskl_max_time
            autoskl_params["max_time_per_run"] = autoskl_max_time_per_run
            autoskl_params["tmp_folder"] = autoskl_tmp_folder
            autoskl_params["output_folder"] = autoskl_output_folder
            autoskl_params["delete_temp"] = True
            autoskl_params["delete_output"] = True
            
            W_train,y_mean_train = weight_matrix_rook(super_H_test)

            X_train, y_train, is_non_hidden_train = basic_feature_matrix(super_H_test)
            lag_train = lagged_feature(super_H_test,W_train,y_mean_train)
            X_train = add_lagged_feature(X_train,lag_train)
            X_train, y_train = remove_hidden(X_train,y_train,is_non_hidden_train) 
            
            flip_hidden(super_H_test)
            
            W_test,y_mean_test = weight_matrix_rook(super_H_test)
            
            X_test, y_test, is_non_hidden_test = basic_feature_matrix(super_H_test)
            lag_test = lagged_feature(super_H_test,W_test,y_mean_test)
            X_test = add_lagged_feature(X_test,lag_test)
            
            error = spatial_lag_auto(X_train,y_train,X_test,y_test,autoskl_params)
            
            with open("/anonymised/path/results/" + experiment_name + "/" + label_set + "/SAR_" + training_set_name + "_" + 
                              test_set_name + ".csv",'a') as fp:
                s = str(None) + "," + str(prop) + "," + str(error) + "\n"
                fp.write(s)

                
