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

from mrp.mrp_misc import *
from mrp.mrp import *
from baselines.general import *






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

bbox_test_bl = bbox_taipei_bl
bbox_test_tr = bbox_taipei_tr


# Shapefiles

shapefile_path_train = "/anonymised/path/shapefiles/Taiwan"
shapefile_path_test = "/anonymised/path/shapefiles/Taiwan"


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











num_runs = 30
ps = [0.1,0.3,0.5,0.7,0.9]
hidden_proportion = 0.8 # Training set

for i in range(0,num_runs):
    for prop in ps:
        
        # These catch-all-exception statements are not great, and optional. We ran long
        # runs on a computing cluster, and did not want random cluster-related crashes
        # to make it awkward to resume the code somewhere in the middle. Running on a PC
        # was no issue without these statements; we merely keep them in because it is what
        # we used to obtain our results.
        while True:
            try:
        
                ############################################
                # Simple regression code
                ############################################
                
                # Set SMAC-optimised preprocessing options where necessary
                
                feature_normalisation_method = "unit"
                missing_value_method = "replace"
                type_filter_method = "top_variable"
                type_top_n_variable = 46

                # Create result files if they do not already exist

                if(not(os.path.exists("/anonymised/path/results/" + experiment_name))):
                    os.mkdir("/anonymised/path/results/" + experiment_name)
                if(not(os.path.exists("/anonymised/path/results/" + experiment_name + "/" + label_set))):
                    os.mkdir("/anonymised/path/results/" + experiment_name + "/" + label_set)
                if(not(os.path.exists("/anonymised/path/results/" + experiment_name + "/" + label_set + "/simple_" + training_set_name + "_" + 
                                      test_set_name + ".csv"))): 
                    with open("/anonymised/path/results/" + experiment_name + "/" + label_set + "/simple_" + training_set_name + "_" + 
                                      test_set_name + ".csv",'w') as fp:
                        fp.write("proportion_train,proportion_test,value\n")



                hidden_proportion_train = hidden_proportion
                hidden_proportion_test = prop
                
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



                # Preprocess training set

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


                # Preprocess test set

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


                # Basic regression code

                # Train

                pred, y_test, m1 = simple_regression_auto(super_H_train,super_H_test,autoskl_params_simple,verbose=False)

                # Test

                MRP = simple_to_MRP(super_H_test,m1,False)
                preds,labels = get_validation_vectors(MRP)

                labels = labels[np.logical_not(np.isnan(preds))]
                preds = preds[np.logical_not(np.isnan(preds))]

                mae_test = mean_absolute_error(labels,preds)

                with open("/anonymised/path/results/" + experiment_name + "/" + label_set + "/simple_" + training_set_name + "_" + 
                                  test_set_name + ".csv",'a') as fp:
                    s = str(hidden_proportion_train) + "," + str(hidden_proportion_test) + "," + str(mae_test) + "\n"
                    fp.write(s)
                    
                break
                
            except:
                pass
    
    
    

        while True:
            try:
    
                ############################################
                # Ordinary kriging 
                ############################################
                
                hidden_proportion_train = hidden_proportion
                hidden_proportion_test = prop
                
                # Preprocess training set

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


                # Preprocess test set

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



                # Find best variogram model on training set

                variogram_models = ["linear", "power", "gaussian", "spherical", "exponential", "hole-effect"]

                best_result = np.inf
                best_var = ""
                for var in variogram_models:
                    loss_test,kriged_grid_test,var_grid_test,label_grid_test,percentage_grid_test,pred_test,y_test,pe_test = baseline_ordinary_kriging(super_H_train,
                                                                                        height_train,width_train,
                                                                                        variogram_model=var,verbose=False)
                    if(loss_test < best_result):
                        best_result = loss_test
                        best_var = var


                # Run on test set

                loss_test,kriged_grid_test,var_grid_test,label_grid_test,percentage_grid_test,pred_test,y_test,pe_test = baseline_ordinary_kriging(super_H_test,
                                                                                        height_test,width_test,
                                                                                        variogram_model=best_var,verbose=False)
                
                
                # Save results
                
                with open("/anonymised/path/results/" + experiment_name + "/" + label_set + "/ordinary_kriging_" + training_set_name + "_" + 
                                  test_set_name + ".csv",'a') as fp:
                    s = str(hidden_proportion_train) + "," + str(hidden_proportion_test) + "," + str(best_result) + "\n"
                    fp.write(s)
                    break
                
            except:
                pass
        
    
    
        while True:
            try:
            
                ############################################
                # Universal kriging
                ############################################
            
            
                # Set up result files
            
                if(not(os.path.exists("/anonymised/path/results/" + experiment_name))):
                    os.mkdir("/anonymised/path/results/" + experiment_name)
                if(not(os.path.exists("/anonymised/path/results/" + experiment_name + "/" + label_set))):
                    os.mkdir("/anonymised/path/results/" + experiment_name + "/" + label_set)
                if(not(os.path.exists("/anonymised/path/results/" + experiment_name + "/" + label_set + "/universal_kriging_" + training_set_name + "_" + 
                                      test_set_name + ".csv"))): 
                    with open("/anonymised/path/results/" + experiment_name + "/" + label_set + "/universal_kriging_" + training_set_name + "_" + 
                                      test_set_name + ".csv",'w') as fp:
                        fp.write("proportion_train,proportion_test,value\n")
                        
                hidden_proportion_train = hidden_proportion
                hidden_proportion_test = prop
                
                # Preprocess training set

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


                # Preprocess test set

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

                # Find best variogram model on training set

                variogram_models = ["linear", "power", "gaussian", "spherical", "exponential", "hole-effect"]

                best_result = np.inf
                best_var = ""
                for var in variogram_models:
                    loss_test,kriged_grid_test,var_grid_test,label_grid_test,percentage_grid_test,pred_test,y_test,pe_test = baseline_universal_kriging(super_H_train,
                                                                                        height_train,width_train,
                                                                                        variogram_model=var,verbose=False)
                    if(loss_test < best_result):
                        best_result = loss_test
                        best_var = var

                # Run on test set

                loss_test,kriged_grid_test,var_grid_test,label_grid_test,percentage_grid_test,pred_test,y_test,pe_test = baseline_universal_kriging(super_H_test,
                                                                                        height_test,width_test,
                                                                                        variogram_model=best_var,verbose=False)
                
                # Save results
                
                with open("/anonymised/path/results/" + experiment_name + "/" + label_set + "/universal_kriging_" + training_set_name + "_" + 
                                  test_set_name + ".csv",'a') as fp:
                    s = str(hidden_proportion_train) + "," + str(hidden_proportion_test) + "," + str(best_result) + "\n"
                    fp.write(s)
                    
                break
                
            except:
                pass
    
    
    
    
        while True:
            try:
    
                ############################################
                # SD-MRP
                ############################################
                
                # Set up result files
                
                if(not(os.path.exists("/anonymised/path/results/" + experiment_name))):
                    os.mkdir("/anonymised/path/results/" + experiment_name)
                if(not(os.path.exists("/anonymised/path/results/" + experiment_name + "/" + label_set))):
                    os.mkdir("/anonymised/path/results/" + experiment_name + "/" + label_set)
                if(not(os.path.exists("/anonymised/path/results/" + experiment_name + "/" + label_set + "/static_discount_" + training_set_name + "_" + 
                                      test_set_name + ".csv"))): 
                    with open("/anonymised/path/results/" + experiment_name + "/" + label_set + "/static_discount_" + training_set_name + "_" + 
                                      test_set_name + ".csv",'w') as fp:
                        fp.write("proportion_train,proportion_test,value\n")
                        
                hidden_proportion_train = hidden_proportion
                hidden_proportion_test = prop
                

                # Preprocess training set

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


                # Preprocess test set

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


                # Parameters, we kept these constant
                epochs = 100
                MRP_iter = 100
                average_loss = True
                
                # Find best gamma on training set

                best_gamma = -1
                best_error = np.inf

                for j in range(0,epochs):
                    MRP_gamma = np.random.uniform(0,1)
                    MRP_final,error,pred = normal_MRP(super_H_train,MRP_iter,MRP_gamma,debug=False)
                    if(error < best_error):
                        best_error = error
                        best_gamma = MRP_gamma


                # Run on test set

                MRP_final_test,mae_test,pred_test = normal_MRP(super_H_test,MRP_iter,best_gamma,debug=False)

                # Save results

                with open("/anonymised/path/results/" + experiment_name + "/" + label_set + "/static_discount_" + training_set_name + "_" + 
                                  test_set_name + ".csv",'a') as fp:
                    s = str(hidden_proportion_train) + "," + str(hidden_proportion_test) + "," + str(mae_test) + "\n"
                    fp.write(s)
                break
                
            except:
                pass
        
    
        while True:
            try:
    
                ############################################
                # O-MRP
                ############################################
            
                # Set up result files
            
                if(not(os.path.exists("/anonymised/path/results/" + experiment_name))):
                    os.mkdir("/anonymised/path/results/" + experiment_name)
                if(not(os.path.exists("/anonymised/path/results/" + experiment_name + "/" + label_set))):
                    os.mkdir("/anonymised/path/results/" + experiment_name + "/" + label_set)
                if(not(os.path.exists("/anonymised/path/results/" + experiment_name + "/" + label_set + "/optimisation_" + training_set_name + "_" + 
                                      test_set_name + ".csv"))): 
                    with open("/anonymised/path/results/" + experiment_name + "/" + label_set + "/optimisation_" + training_set_name + "_" + 
                                      test_set_name + ".csv",'w') as fp:
                        fp.write("proportion_train,proportion_test,value\n")
                        
                hidden_proportion_train = hidden_proportion
                hidden_proportion_test = prop
                

                # Preprocess training set

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


                # Preprocess test set

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


                # Call optimisation function

                MRP_final_test,mae_test,W = optimise_weights(super_H_test,optimisation_epochs,MRP_iter,mutation_intensity)
                
                # Save results
                
                with open("/anonymised/path/results/" + experiment_name + "/" + label_set + "/optimisation_" + training_set_name + "_" + 
                                  test_set_name + ".csv",'a') as fp:
                    s = str(hidden_proportion_train) + "," + str(hidden_proportion_test) + "," + str(mae_test) + "\n"
                    fp.write(s)
                    
                break
                
            except:
                pass
    
    
        while True:
            try:
    
                ############################################
                # WP-MRP
                ############################################
            
                # Custom preprocessing and hyperparameters as per SMAC results
            
                MRP_iter = 115
                feature_normalisation_method = "none"
                missing_value_method = "drop"
                mutation_intensity = 0.9658169205831775
                optimisation_epochs = 79
                train_proportion = 0.34555591941979796
                type_filter_method = "taxonomy"
                
                # Set up result files
                
                if(not(os.path.exists("/anonymised/path/results/" + experiment_name))):
                    os.mkdir("/anonymised/path/results/" + experiment_name)
                if(not(os.path.exists("/anonymised/path/results/" + experiment_name + "/" + label_set))):
                    os.mkdir("/anonymised/path/results/" + experiment_name + "/" + label_set)
                if(not(os.path.exists("/anonymised/path/results/" + experiment_name + "/" + label_set + "/weights_" + training_set_name + "_" + 
                                      test_set_name + ".csv"))): 
                    with open("/anonymised/path/results/" + experiment_name + "/" + label_set + "/weights_" + training_set_name + "_" + 
                                      test_set_name + ".csv",'w') as fp:
                        fp.write("proportion_train,proportion_test,value\n")
                        
                hidden_proportion_train = hidden_proportion
                hidden_proportion_test = prop
                
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

                # Preprocess training set

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


                # Preprocess test set

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


                # Train model

                MRP = normalise_attributes(super_H_train,feature_normalisation_method)
                model, pred_loss, opt_loss = supervise_by_optimisation_auto(MRP,MRP_iter,optimisation_epochs,
                                                                  mutation_intensity,train_proportion,autoskl_params_weights,
                                                                       average_loss)

                # Run on test set

                MRP = normalise_attributes(super_H_test,feature_normalisation_method)
                MRP_final_test,mae_test,pred_test = run_MRP_with_model(MRP,model,MRP_iter)
                
                # Save results
                
                with open("/anonymised/path/results/" + experiment_name + "/" + label_set + "/weights_" + training_set_name + "_" + 
                                  test_set_name + ".csv",'a') as fp:
                    s = str(hidden_proportion_train) + "," + str(hidden_proportion_test) + "," + str(mae_test) + "\n"
                    fp.write(s)
    
                break
                
            except:
                pass
    
    
    
    
    
    
    
    

                
