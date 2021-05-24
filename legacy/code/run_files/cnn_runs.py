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

import pandas as pd
import numpy as np
import networkx as nx

from sklearn.metrics import mean_absolute_error as mae
import autokeras as ak
from tensorflow.keras.models import load_model
import tensorflow as tf

import os



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

region_size_lat = 0.1 # Degrees on map; not used in practice
region_size_lon = 0.1 # Degrees on map; not used in practice
location_size_lat = 0.02 # Degrees on map
location_size_lon = 0.02 # Degrees on map

region_params = [region_size_lat,region_size_lon,location_size_lat,location_size_lon]

# MRP parameters

hidden_proportion = 0.8


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
  
    


# Variable parameters


# Type parameters

type_frequency_ratio = 0.5887024334058893
type_top_n = 10
type_top_n_percent = 20
type_top_n_variable = 15

type_params = [type_frequency_ratio,type_top_n,type_top_n_percent,type_top_n_variable]

region_min_objects = 0



# Methods

# Possible values: "replace","drop"
missing_value_method = "replace"

# Possible values: "frequency,top,top_percent,top_variable,taxonomy,none"
type_filter_method = "frequency"

# Possible values: "unit","z_score","mean_norm","none"
feature_normalisation_method = "none"



# CNN parameters

nn_window_height = 17
nn_window_width = 18
nn_validation_split = 0.33467051772273004


# Run ID

run_id = 1
if(not(os.path.exists("/anonymised/path/results/neural_network/id.txt"))): 
    with open("/anonymised/path/results/neural_network/id.txt",'w') as fp:
        fp.write("2")
else:
    with open("/anonymised/path/results/neural_network/id.txt",'r') as fp:
        s = fp.read()
        run_id = int(s)
    with open("/anonymised/path/results/neural_network/id.txt",'w') as fp:
        fp.write(str(run_id+1))

# Practical

num_runs = 30
proportions = [0.1,0.3,0.5,0.7,0.9]



    
    

##################
# Taichung-Taipei
##################

experiment_name = "CNN_Taichung_Taipei"
shapefile_path_train = taichung_dict['shp']
bbox_train_bl = taichung_dict['bbox_bl']
bbox_train_tr = taichung_dict['bbox_tr']

shapefile_path_test = taipei_dict['shp']
bbox_test_bl = taipei_dict['bbox_bl']
bbox_test_tr = taipei_dict['bbox_tr']



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



hidden_proportion_train = 0.8
hidden_proportion_test = 0.5 # This one is kind of hacky, it gets changed in the loop anyway, but this way
                                # I can use the same preprocessing function for both


# Prepare result files

if(not(os.path.exists("/anonymised/path/results/" + experiment_name))):
    os.mkdir("/anonymised/path/results/" + experiment_name)
if(not(os.path.exists("/anonymised/path/results/" + experiment_name + "/" + label_set))):
    os.mkdir("/anonymised/path/results/" + experiment_name + "/" + label_set)
if(not(os.path.exists("/anonymised/path/results/" + experiment_name + "/" + label_set + "/CNN_" + training_set_name + "_" + 
                      test_set_name + ".csv"))): 
    with open("/anonymised/path/results/" + experiment_name + "/" + label_set + "/CNN_" + training_set_name + "_" + 
                      test_set_name + ".csv",'w') as fp:
        fp.write("proportion_train,proportion_test,value\n")


        


# Preprocess train+test set

S = load_spatial_data(shapefile_path_train,missing_value_method)
S = clip_area(S,bbox_train_bl,bbox_train_tr)

S,types = find_types(S,optimisation_path,working_path,type_filter_method,type_params,
                     taxonomy_filename=taxonomy_filename,verbose=False)

S = compute_centroids(S)
region_bounds = compute_region_bounds(S,location_size_lat,location_size_lon)
regions,region_bounds = assign_objects_to_regions(S,region_bounds,region_min_objects=region_min_objects)
super_G = create_super_graph_raw(regions,region_bounds,types,location_size_lat,location_size_lon)
super_H_train,width_train,height_train = convert_super_G(super_G,S,label_paths[label_ind],region_params,hidden_proportion_train,
                                                                    from_grid=label_from_grid)



S = load_spatial_data(shapefile_path_test,missing_value_method)
S = clip_area(S,bbox_test_bl,bbox_test_tr)

S,types = find_types(S,optimisation_path,working_path,type_filter_method,type_params,
                     taxonomy_filename=taxonomy_filename,verbose=False)

S = compute_centroids(S)
region_bounds = compute_region_bounds(S,location_size_lat,location_size_lon)
regions,region_bounds = assign_objects_to_regions(S,region_bounds,region_min_objects=region_min_objects)
super_G = create_super_graph_raw(regions,region_bounds,types,location_size_lat,location_size_lon)
super_H_test,width_test,height_test = convert_super_G(super_G,S,label_paths[label_ind],region_params,hidden_proportion_test,
                                                                    from_grid=label_from_grid)
        
        
        
        
for i in range(0,num_runs):
    print("Run #",i)
    
    for hidden_proportion in proportions:

        # Shuffle hidden nodes

        hidden_proportion_test = hidden_proportion
        super_H_train = shuffle_hidden(super_H_train,hidden_proportion_train)
        super_H_test = shuffle_hidden(super_H_test,hidden_proportion)

        # Prepare data for CNN

        X_train, y_train = graph_to_tensor_train(super_H_train,nn_window_height,nn_window_width)
        X_test, y_test = graph_to_tensor_test(super_H_test,nn_window_height,nn_window_width)
        
        # Load optimised model; we will re-train but this way the architecture is
        # exactly as optimised
        
        model_original = load_model("/anonymised/path/autokeras/models/gdp_taichung_daegu_50", 
                          custom_objects=ak.CUSTOM_OBJECTS)

        model = tf.keras.models.clone_model(model_original)
        model.compile(loss="mean_absolute_error")
        
        # Re-train
        
        model.fit(X_train, y_train, validation_split=nn_validation_split)
        
        # Compute error
        
        pred = model.predict(X_test)
        error = mae(pred,y_test)

        # Save results

        with open("/anonymised/path/results/" + experiment_name + "/" + label_set + "/CNN_" + training_set_name + "_" + 
                          test_set_name + ".csv",'a') as fp:
            s = str(hidden_proportion_train) + "," + str(hidden_proportion) + "," + str(error) + "\n"
            fp.write(s)







##################
# Taichung-Daegu
##################

experiment_name = "CNN_Taichung_Taipei"
shapefile_path_train = taichung_dict['shp']
bbox_train_bl = taichung_dict['bbox_bl']
bbox_train_tr = taichung_dict['bbox_tr']

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



hidden_proportion_train = 0.8
hidden_proportion_test = 0.5 # This one is kind of hacky, it gets changed in the loop anyway, but this way
                                # I can use the same preprocessing function for both


if(not(os.path.exists("/anonymised/path/results/" + experiment_name))):
    os.mkdir("/anonymised/path/results/" + experiment_name)
if(not(os.path.exists("/anonymised/path/results/" + experiment_name + "/" + label_set))):
    os.mkdir("/anonymised/path/results/" + experiment_name + "/" + label_set)
if(not(os.path.exists("/anonymised/path/results/" + experiment_name + "/" + label_set + "/CNN_" + training_set_name + "_" + 
                      test_set_name + ".csv"))): 
    with open("/anonymised/path/results/" + experiment_name + "/" + label_set + "/CNN_" + training_set_name + "_" + 
                      test_set_name + ".csv",'w') as fp:
        fp.write("proportion_train,proportion_test,value\n")


        


# Preprocess train+test set

S = load_spatial_data(shapefile_path_train,missing_value_method)
S = clip_area(S,bbox_train_bl,bbox_train_tr)

S,types = find_types(S,optimisation_path,working_path,type_filter_method,type_params,
                     taxonomy_filename=taxonomy_filename,verbose=False)

S = compute_centroids(S)
region_bounds = compute_region_bounds(S,location_size_lat,location_size_lon)
regions,region_bounds = assign_objects_to_regions(S,region_bounds,region_min_objects=region_min_objects)
super_G = create_super_graph_raw(regions,region_bounds,types,location_size_lat,location_size_lon)
super_H_train,width_train,height_train = convert_super_G(super_G,S,label_paths[label_ind],region_params,hidden_proportion_train,
                                                                    from_grid=label_from_grid)



S = load_spatial_data(shapefile_path_test,missing_value_method)
S = clip_area(S,bbox_test_bl,bbox_test_tr)

S,types = find_types(S,optimisation_path,working_path,type_filter_method,type_params,
                     taxonomy_filename=taxonomy_filename,verbose=False)

S = compute_centroids(S)
region_bounds = compute_region_bounds(S,location_size_lat,location_size_lon)
regions,region_bounds = assign_objects_to_regions(S,region_bounds,region_min_objects=region_min_objects)
super_G = create_super_graph_raw(regions,region_bounds,types,location_size_lat,location_size_lon)
super_H_test,width_test,height_test = convert_super_G(super_G,S,label_paths[label_ind],region_params,hidden_proportion_test,
                                                                    from_grid=label_from_grid)
        
        
        
        
for i in range(0,num_runs):
    print("Run #",i)
    
    for hidden_proportion in proportions:

        hidden_proportion_test = hidden_proportion
        super_H_train = shuffle_hidden(super_H_train,hidden_proportion_train)
        super_H_test = shuffle_hidden(super_H_test,hidden_proportion)

        X_train, y_train = graph_to_tensor_train(super_H_train,nn_window_height,nn_window_width)
        X_test, y_test = graph_to_tensor_test(super_H_test,nn_window_height,nn_window_width)
        
        model_original = load_model("/anonymised/path/autokeras/models/gdp_taichung_daegu_50", 
                          custom_objects=ak.CUSTOM_OBJECTS)

        model = tf.keras.models.clone_model(model_original)
        model.compile(loss="mean_absolute_error")
        
        model.fit(X_train, y_train, validation_split=nn_validation_split)
        
        pred = model.predict(X_test)
        error = mae(pred,y_test)


        with open("/anonymised/path/results/" + experiment_name + "/" + label_set + "/CNN_" + training_set_name + "_" + 
                          test_set_name + ".csv",'a') as fp:
            s = str(hidden_proportion_train) + "," + str(hidden_proportion) + "," + str(error) + "\n"
            fp.write(s)




