from utils.generate_data import *
from utils.hide_spatial_data import *
from utils.experiments import run_experiments_2D, run_experiments_3D
from utils.load_shapefile_features import *
from utils.load_trajectories import *
from utils.load_raster import *
from utils.load_csv_data import *
from utils.load_msi import *

from sklearn.linear_model import LinearRegression
import autosklearn.regression
import autokeras as ak

from MRPinterpolation.SD_MRP import SD_SMRP, SD_STMRP
from MRPinterpolation.WP_MRP import WP_SMRP, WP_STMRP


# General

setting_name = "synthetic_2D"

# Valid: "porto_trajectories", "synthetic_2D", "synthetic_3D", "GDP", "covid", "satellites"
dataset = "synthetic_2D"
alg = "SD_MRP"
save_dir = "results"
save_path = "/home/laurens/Projects/MRPinterpolation"
hidden_method = "random"

num_trials = 2

hidden_proportion = 0.8

num_points = 5
radius =  4
num_traj = 10

taxonomy_path = "/mnt/e/User Files/Projects/Master Thesis/Working/type_taxonomy_v1.tsv"


# SMAC

max_runtime = 1 # hours
max_memory_per_run = 5 # GB



# Trajectories

traj_path_original = "/mnt/c/Users/Laurens/Projects/Porto Trajectories/porto_trajectories_all.csv"
traj_path = "/mnt/c/Users/Laurens/Projects/Porto Trajectories/porto_trajectories_preprocessed.csv"

#num_timesteps = 4

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

traj_variable = "sources" # sources, targets


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

gdp_path = "/mnt/e/User Files/Projects/Master Thesis/Label grids/GDP/GDP.tif"

gdp_bbox_tl = (121.3485,25.2465)
gdp_bbox_br = (121.7760,24.8192)

gdp_spatial_path = "/mnt/e/User Files/Projects/Master Thesis/Taiwan fewer shapefiles/"
gdp_spatial_sources = ["gis_osm_buildings_a_free_1.shp",
                   "gis_osm_natural_free_1.shp",
                  "gis_osm_places_free_1.shp",
                  "gis_osm_pofw_free_1.shp",
                  "gis_osm_traffic_a_free_1.shp",
                  "gis_osm_transport_a_free_1.shp"]

gdp_spatial_data = [(gdp_spatial_path + s) for s in gdp_spatial_sources]


# Covid

covid_path = "/mnt/e/User Files/Projects/Master Thesis/Label grids/Covid Korea/PatientRoute.csv"

covid_bbox_tl = (35.9772,128.4298)
covid_bbox_br = (35.7642,128.7956)

covid_res_y = 100
covid_res_x = 100

covid_spatial_path = "/mnt/e/User Files/Projects/Master Thesis/Korea fewer shapefiles/"
covid_spatial_sources = ["gis_osm_buildings_a_free_1.shp",
                   "gis_osm_natural_free_1.shp",
                  "gis_osm_places_free_1.shp",
                  "gis_osm_pofw_free_1.shp",
                  "gis_osm_traffic_a_free_1.shp",
                  "gis_osm_transport_a_free_1.shp"]

covid_spatial_data = [(covid_spatial_path + s) for s in covid_spatial_sources]

# Satellites

sat_most_recent_path = "/mnt/c/Users/Laurens/Projects/Satellite Data/Mediterranean/Crops/GeoTIFF/most_recent.tif"
sat_week_1_path = "/mnt/c/Users/Laurens/Projects/Satellite Data/Mediterranean/Crops/GeoTIFF/1_week.tif"
sat_month_1_path = "/mnt/c/Users/Laurens/Projects/Satellite Data/Mediterranean/Crops/GeoTIFF/1_month.tif"
sat_month_6_path = "/mnt/c/Users/Laurens/Projects/Satellite Data/Mediterranean/Crops/GeoTIFF/6_months.tif"
sat_month_12_path = "/mnt/c/Users/Laurens/Projects/Satellite Data/Mediterranean/Crops/GeoTIFF/12_months.tif"

paths = [sat_most_recent_path,sat_week_1_path,sat_month_1_path,sat_month_6_path,sat_month_12_path]




cfg = {}

if(dataset == "synthetic_2D"):
    # Data preprocessing stuff
    
    # Algorithm configs synthetic_2D
    if(alg == "SD_MRP"):
        cfg["MRP_iterations"] = 50
        cfg["MRP_SD_epochs"] = 100
        cfg["MRP_sub_iterations"] = 100
        cfg["MRP_subsample_proportion"] = 0.5

    elif(alg == "WP_MRP"):
        cfg["MRP_iterations"] = 50

    elif(alg == "OK"):
        cfg["OK_variogram"] = "linear"

    elif(alg == "UK"):
        cfg["UK_variogram"] = "linear"

    elif(alg == "CNN"):
        cfg["nn_max_trials"] = 5
        cfg["nn_epochs"] = 5
        cfg["nn_train_fill"] = "True"
        cfg["nn_window_height"] = 5
        cfg["nn_window_width"] = 5
    
elif(dataset == "synthetic_3D"):
    # Data preprocessing stuff
    
    # Algorithm configs synthetic_2D
    if(alg == "SD_MRP"):
        cfg["MRP_iterations"] = 50
        cfg["MRP_SD_epochs"] = 100
        cfg["MRP_sub_iterations"] = 100
        cfg["MRP_subsample_proportion"] = 0.5

    elif(alg == "WP_MRP"):
        cfg["MRP_iterations"] = 50

    elif(alg == "OK"):
        cfg["OK_variogram"] = "linear"

    elif(alg == "UK"):
        cfg["UK_variogram"] = "linear"

    elif(alg == "CNN"):
        cfg["nn_max_trials"] = 5
        cfg["nn_epochs"] = 5
        cfg["nn_train_fill"] = "True"
        cfg["nn_window_height"] = 5
        cfg["nn_window_width"] = 5

elif(dataset == "satellites"):
    # Data preprocessing stuff
    
    # Algorithm configs synthetic_2D
    if(alg == "SD_MRP"):
        cfg["MRP_iterations"] = 50
        cfg["MRP_SD_epochs"] = 100
        cfg["MRP_sub_iterations"] = 100
        cfg["MRP_subsample_proportion"] = 0.5

    elif(alg == "WP_MRP"):
        cfg["MRP_iterations"] = 50

    elif(alg == "OK"):
        cfg["OK_variogram"] = "linear"

    elif(alg == "UK"):
        cfg["UK_variogram"] = "linear"

    elif(alg == "CNN"):
        cfg["nn_max_trials"] = 5
        cfg["nn_epochs"] = 5
        cfg["nn_train_fill"] = "True"
        cfg["nn_window_height"] = 5
        cfg["nn_window_width"] = 5

elif(dataset == "porto_trajectories"):
    # Data preprocessing stuff
    
    cfg["missing_value_method"] = "drop"
    cfg["type_filter_method"] = "taxonomy"
    cfg["feature_normalisation_method"] = "mean_norm"
    cfg["type_top_frequent"] = 10
    
    # Algorithm configs synthetic_2D
    if(alg == "SD_MRP"):
        cfg["MRP_iterations"] = 50
        cfg["MRP_SD_epochs"] = 100
        cfg["MRP_sub_iterations"] = 100
        cfg["MRP_subsample_proportion"] = 0.5

    elif(alg == "WP_MRP"):
        cfg["MRP_iterations"] = 50

    elif(alg == "OK"):
        cfg["OK_variogram"] = "linear"

    elif(alg == "UK"):
        cfg["UK_variogram"] = "linear"

    elif(alg == "CNN"):
        cfg["nn_max_trials"] = 5
        cfg["nn_epochs"] = 5
        cfg["nn_train_fill"] = "True"
        cfg["nn_window_height"] = 5
        cfg["nn_window_width"] = 5

elif(dataset == "GDP"):
    # Data preprocessing stuff
    
    cfg["missing_value_method"] = "drop"
    cfg["type_filter_method"] = "taxonomy"
    cfg["feature_normalisation_method"] = "mean_norm"
    cfg["type_top_frequent"] = 10
    
    # Algorithm configs synthetic_2D
    if(alg == "SD_MRP"):
        cfg["MRP_iterations"] = 50
        cfg["MRP_SD_epochs"] = 100
        cfg["MRP_sub_iterations"] = 100
        cfg["MRP_subsample_proportion"] = 0.5

    elif(alg == "WP_MRP"):
        cfg["MRP_iterations"] = 50

    elif(alg == "OK"):
        cfg["OK_variogram"] = "linear"

    elif(alg == "UK"):
        cfg["UK_variogram"] = "linear"

    elif(alg == "CNN"):
        cfg["nn_max_trials"] = 5
        cfg["nn_epochs"] = 5
        cfg["nn_train_fill"] = "True"
        cfg["nn_window_height"] = 5
        cfg["nn_window_width"] = 5

elif(dataset == "covid"):
    # Data preprocessing stuff
    
    cfg["missing_value_method"] = "drop"
    cfg["type_filter_method"] = "taxonomy"
    cfg["feature_normalisation_method"] = "mean_norm"
    cfg["type_top_frequent"] = 10
    
    # Algorithm configs synthetic_2D
    if(alg == "SD_MRP"):
        cfg["MRP_iterations"] = 50
        cfg["MRP_SD_epochs"] = 100
        cfg["MRP_sub_iterations"] = 100
        cfg["MRP_subsample_proportion"] = 0.5

    elif(alg == "WP_MRP"):
        cfg["MRP_iterations"] = 50

    elif(alg == "OK"):
        cfg["OK_variogram"] = "linear"

    elif(alg == "UK"):
        cfg["UK_variogram"] = "linear"

    elif(alg == "CNN"):
        cfg["nn_max_trials"] = 5
        cfg["nn_epochs"] = 5
        cfg["nn_train_fill"] = "True"
        cfg["nn_window_height"] = 5
        cfg["nn_window_width"] = 5


        
        
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
        # TODO: load model per dataset
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
        
    params["save_dir"] = save_dir
    params["save_path"] = save_path
    params["setting_name"] = setting_name
    params["alg"] = alg
    params["hidden_method"] = hidden_method
    
    if(dataset == "porto_trajectories" or dataset == "synthetic_3D"):
        result_grids, runtimes = run_experiments_3D(grid,f_grid,alg,num_trials,
                                                        params,hidden_method=hidden_method,save=True)
        result = result_grids[0,:,:,:]
    else:
        result_grids, runtimes = run_experiments_2D(grid,f_grid,alg,num_trials,
                                                        params,hidden_method=hidden_method,save=True)
        result = result_grids[0,:,:]
    error = np.mean(np.absolute(result-grid))
    return(error)
    
    
def run_trajectories(cfg):

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
        
    if(traj_variable == "sources"):
        error = run_general(grid[:,:,:,0],f_grid,cfg)
    else:
        error = run_general(grid[:,:,:,1],f_grid,cfg)

def run_synthetic_2D(cfg):
    # Preprocess
    
    grid, f_grid = generate_data(user_params=synth2d_params,generate_features=True)
    error = run_general(grid,f_grid,cfg)

def run_synthetic_3D(cfg):
    # Preprocess
    
    grid, f_grid = generate_3D_data(user_params=synth3d_params,generate_features=True)
    error = run_general(grid,f_grid,cfg)
    
def run_GDP(cfg):
    grid = raster_to_grid(gdp_path,gdp_bbox_tl,gdp_bbox_br)
    
    if(alg != "OK" and  alg != "UK" and alg != "SD_MRP"):
        additional_params = {}
        if(cfg["type_filter_method"] == "taxonomy"):
            additional_params["taxonomy_path"] = taxonomy_path
        elif(cfg["type_filter_method"] == "top_frequent"):
            additional_params["num_features"] = cfg["type_top_frequent"]

        meta = get_meta(grid.shape[0],grid.shape[1],gdp_bbox_tl,gdp_bbox_br)
        S = load_spatial_data(gdp_spatial_data,gdp_bbox_tl,gdp_bbox_br,cfg["missing_value_method"])
        f_grid = assign_shapes_to_f_grid(S,meta,cfg["type_filter_method"],additional_params=additional_params)
        f_grid = normalise_attributes(f_grid,cfg["feature_normalisation_method"])
    else:
        f_grid = None
    
    error = run_general(grid,f_grid,cfg)

def run_covid(cfg):
    if(alg != "OK" and  alg != "UK" and alg != "SD_MRP"):
        additional_params = {}
        if(cfg["type_filter_method"] == "taxonomy"):
            additional_params["taxonomy_path"] = taxonomy_path
        elif(cfg["type_filter_method"] == "top_frequent"):
            additional_params["num_features"] = cfg["type_top_frequent"]

    # Preprocess
    
    df = pd.read_csv(covid_path)
    df = filter_bbox_csv(df,covid_bbox_tl,covid_bbox_br)
    meta = get_meta(covid_res_y,covid_res_x,covid_bbox_tl,covid_bbox_br)
    grid = assign_csv_to_grid(df,meta)
    f_grid = None

    if(alg != "OK" and  alg != "UK" and alg != "SD_MRP"):
        S = load_spatial_data(covid_spatial_data,covid_bbox_tl,covid_bbox_br,cfg["missing_value_method"])
        f_grid = assign_shapes_to_f_grid(S,meta,cfg["type_filter_method"],additional_params=additional_params)
        f_grid = normalise_attributes(f_grid,cfg["feature_normalisation_method"])

    error = run_general(grid,f_grid,cfg)

def run_satellites(cfg):
    # Preprocess
    
    data = load_satellite_data(paths,normalise=True)
    grid, f_grid = msi_to_grid(data,target_index=0,band_index=0,generate_features=True)
    
    error = run_general(grid,f_grid,cfg)
    
    
    
if(dataset == "porto_trajectories"):
    run_trajectories(cfg)
elif(dataset == "covid"):
    run_covid(cfg)
elif(dataset == "GDP"):
    run_GDP(cfg)
elif(dataset == "satellites"):
    run_satellites(cfg)
elif(dataset == "synthetic_2D"):
    run_synthetic_2D(cfg)
elif(dataset == "synthetic_3D"):
    run_synthetic_3D(cfg)
