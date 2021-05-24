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

num_runs = 15
ps = [0.1,0.3,0.5,0.7,0.9]
cities = [seoul_dict,daegu_dict,taipei_dict,taichung_dict]

hidden_proportion = 0.8 




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
    # Static discount
    ############################################

    # Setup result file
    if(not(os.path.exists("/anonymised/path/results/" + experiment_name))):
        os.mkdir("/anonymised/path/results/" + experiment_name)
    if(not(os.path.exists("/anonymised/path/results/" + experiment_name + "/" + label_set))):
        os.mkdir("/anonymised/path/results/" + experiment_name + "/" + label_set)
    if(not(os.path.exists("/anonymised/path/results/" + experiment_name + "/" + label_set + "/static_discount_" + training_set_name + "_" + 
                          test_set_name + ".csv"))): 
        with open("/anonymised/path/results/" + experiment_name + "/" + label_set + "/static_discount_" + training_set_name + "_" + 
                          test_set_name + ".csv",'w') as fp:
            fp.write("proportion_train,proportion_test,value\n")


    # Practical variables

    epochs = 100
    MRP_iter = 100
    average_loss = True

    # Preprocess test set

    S = load_spatial_data(shapefile_path_test,missing_value_method)
    S = clip_area(S,bbox_test_bl,bbox_test_tr)

    S,types = find_types(S,optimisation_path,working_path,type_filter_method,type_params,
                         taxonomy_filename=taxonomy_filename,verbose=False)

    S = compute_centroids(S)
    region_bounds = compute_region_bounds(S,location_size_lat,location_size_lon)
    regions,region_bounds = assign_objects_to_regions(S,region_bounds,region_min_objects=region_min_objects)
    super_G = create_super_graph_raw(regions,region_bounds,types,location_size_lat,location_size_lon)

    super_H_test,width_test,height_test = convert_super_G(super_G,S,label_paths[label_ind],region_params,hidden_proportion,from_grid=label_from_grid)



    for prop in ps:
                                                                               
        for i in range(0,num_runs):
        
            super_H_test = shuffle_hidden(super_H_test,prop)
        
          
            # Actual code
            
            MRP_final_test,mae_test,pred_test = normal_MRP_self_supervised(super_H_test,MRP_iter,epochs,0.5)


            with open("/anonymised/path/results/" + experiment_name + "/" + label_set + "/static_discount_" + training_set_name + "_" + 
                              test_set_name + ".csv",'a') as fp:
                s = str(None) + "," + str(prop) + "," + str(mae_test) + "\n"
                fp.write(s)

    ############################################
    # Weight prediction
    ############################################

    # Setup result file
    if(not(os.path.exists("/anonymised/path/results/" + experiment_name))):
        os.mkdir("/anonymised/path/results/" + experiment_name)
    if(not(os.path.exists("/anonymised/path/results/" + experiment_name + "/" + label_set))):
        os.mkdir("/anonymised/path/results/" + experiment_name + "/" + label_set)
    if(not(os.path.exists("/anonymised/path/results/" + experiment_name + "/" + label_set + "/static_discount_" + training_set_name + "_" + 
                          test_set_name + ".csv"))): 
        with open("/anonymised/path/results/" + experiment_name + "/" + label_set + "/static_discount_" + training_set_name + "_" + 
                          test_set_name + ".csv",'w') as fp:
            fp.write("proportion_train,proportion_test,value\n")


    # Preprocessing variables

    MRP_iter = 115
    feature_normalisation_method = "none"
    missing_value_method = "drop"
    mutation_intensity = 0.9658169205831775
    optimisation_epochs = 79
    train_proportion = 0.34555591941979796
    type_filter_method = "taxonomy"

    # Preprocess test set

    S = load_spatial_data(shapefile_path_test,missing_value_method)
    S = clip_area(S,bbox_test_bl,bbox_test_tr)

    S,types = find_types(S,optimisation_path,working_path,type_filter_method,type_params,
                         taxonomy_filename=taxonomy_filename,verbose=False)

    S = compute_centroids(S)
    region_bounds = compute_region_bounds(S,location_size_lat,location_size_lon)
    regions,region_bounds = assign_objects_to_regions(S,region_bounds,region_min_objects=region_min_objects)
    super_G = create_super_graph_raw(regions,region_bounds,types,location_size_lat,location_size_lon)

    super_H_test,width_test,height_test = convert_super_G(super_G,S,label_paths[label_ind],region_params,prop,from_grid=label_from_grid)



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
                    
            autoskl_tmp_folder_weights = "/anonymised/path/autosklearn/weights/" + str(autoskl_current_id) + "temp"
            autoskl_output_folder_weights = "/anonymised/path/autosklearn/weights/" + str(autoskl_current_id) + "out"
            
            autoskl_params_weights["tmp_folder"] = autoskl_tmp_folder_weights
            autoskl_params_weights["output_folder"] = autoskl_output_folder_weights
        
          
            # Actual code
            
            # Train
            MRP = normalise_attributes(super_H_test,feature_normalisation_method)
            model, pred_loss, opt_loss = supervise_by_optimisation_auto(MRP,MRP_iter,optimisation_epochs,mutation_intensity,train_proportion,autoskl_params_weights,
                                                                   average_loss)
                                                                   
            # Test
            flip_hidden(super_H_test)
            MRP_final_test,mae_test,pred_test = run_MRP_with_model(MRP,model,MRP_iter)
            
            with open("/anonymised/path/results/" + experiment_name + "/" + label_set + "/weights_" + training_set_name + "_" + 
                              test_set_name + ".csv",'a') as fp:
                s = str(None) + "," + str(prop) + "," + str(mae_test) + "\n"
                fp.write(s)




    ############################################
    # Simple regression 
    ############################################

    # Setup result file
    if(not(os.path.exists("/anonymised/path/results/" + experiment_name))):
        os.mkdir("/anonymised/path/results/" + experiment_name)
    if(not(os.path.exists("/anonymised/path/results/" + experiment_name + "/" + label_set))):
        os.mkdir("/anonymised/path/results/" + experiment_name + "/" + label_set)
    if(not(os.path.exists("/anonymised/path/results/" + experiment_name + "/" + label_set + "/simple_" + training_set_name + "_" + 
                          test_set_name + ".csv"))): 
        with open("/anonymised/path/results/" + experiment_name + "/" + label_set + "/simple_" + training_set_name + "_" + 
                          test_set_name + ".csv",'w') as fp:
            fp.write("proportion_train,proportion_test,value\n")


    # Preprocessing variables

    feature_normalisation_method = "unit"
    missing_value_method = "replace"
    type_filter_method = "top_variable"
    type_top_n_variable = 46

    # Preprocess test set

    S = load_spatial_data(shapefile_path_test,missing_value_method)
    S = clip_area(S,bbox_test_bl,bbox_test_tr)

    S,types = find_types(S,optimisation_path,working_path,type_filter_method,type_params,
                         taxonomy_filename=taxonomy_filename,verbose=False)

    S = compute_centroids(S)
    region_bounds = compute_region_bounds(S,location_size_lat,location_size_lon)
    regions,region_bounds = assign_objects_to_regions(S,region_bounds,region_min_objects=region_min_objects)
    super_G = create_super_graph_raw(regions,region_bounds,types,location_size_lat,location_size_lon)

    super_H_test,width_test,height_test = convert_super_G(super_G,S,label_paths[label_ind],region_params,prop,
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
                    
            autoskl_tmp_folder_simple = "/anonymised/path/autosklearn/simple/" + str(autoskl_current_id) + "temp"
            autoskl_output_folder_simple = "/anonymised/path/autosklearn/simple/" + str(autoskl_current_id) + "out"
            
            autoskl_params_simple["tmp_folder"] = autoskl_tmp_folder_simple
            autoskl_params_simple["output_folder"] = autoskl_output_folder_simple
            
            # Actual code

            # Train
            m1 = simple_regression_auto_self(super_H_test,autoskl_params_simple)

            # Test 
            flip_hidden(super_H_test)
            MRP = simple_to_MRP(super_H_test,m1,False)

            preds,labels = get_validation_vectors(MRP)

            labels = labels[np.logical_not(np.isnan(preds))]
            preds = preds[np.logical_not(np.isnan(preds))]

            mae_test = mean_absolute_error(labels,preds)

            with open("/anonymised/path/results/" + experiment_name + "/" + label_set + "/simple_" + training_set_name + "_" + 
                              test_set_name + ".csv",'a') as fp:
                s = str(None) + "," + str(prop) + "," + str(mae_test) + "\n"
                fp.write(s)


