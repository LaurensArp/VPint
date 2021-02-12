import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'../'))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR

from MRPinterpolation.SMRP import SD_SMRP, WP_SMRP
from utils.generate_data_spatial import *

# Parameters

# Grid

grid_width = 20
grid_height = 20

# Synthetic data generation

param_global_mean = 20
param_global_std = 10

param_stationary = False
param_nonstationarity_num_points = 10
param_nonstationary_min_mean = 10
param_nonstationary_max_mean = 100

param_ac = 0.5
param_ac_iterations = 3
param_ac_static = True
param_ac_mean = 0.5
param_ac_std = 0.3
ac_params = {
    "static":param_ac_static,
    "autocorrelation":param_ac,
    "mean":param_ac_mean,
    "std":param_ac_std,
    "iterations":param_ac_iterations
}

param_feature_min = 0
param_feature_max = 1
param_num_features = 1
param_feature_correlation = 0.5

feature_params = {
    "num":param_num_features,
    "min":param_feature_min,
    "max":param_feature_max,
    "correlation":param_feature_correlation,
}

param_isotropy = True
param_anisotropy_coeff = {
    "left":0.2,
    "right":0.5,
    "top":0.2,
    "down":0.5
}

# MRP

param_hidden_prob = 0.5
param_SD_search_epochs = 100
param_MRP_iter = 100
param_train_prop = 0.5


# Generate grids

nonstationary_points = generate_nonstationary_points(param_nonstationarity_num_points,
                grid_height,grid_width,param_nonstationary_min_mean,param_nonstationary_max_mean)
grid = create_grid(grid_height,grid_width,param_global_mean,param_global_std,
                stationary=param_stationary,nonstationary_points=nonstationary_points)
grid = update_grid(grid,ac_params,grid_height,grid_width,
                isotropy=param_isotropy)
feature_grid = assign_features(grid,feature_params)

# Hide values

hidden_grid = hide_values_uniform(grid,param_hidden_prob)

# Run SD-MRP

MRP = SD_SMRP(hidden_grid)
MRP.find_gamma(param_SD_search_epochs,param_train_prop)
pred_grid_SD = MRP.run(param_MRP_iter)

# Compute MAE, visualise pixel error

mae, mae_grid = MRP.mean_absolute_error(grid,gridded=True)
print("MAE SD-SMRP: " + str(mae))
plt.imshow(mae_grid)
plt.title("SD-SMRP Absolute error per pixel")
plt.show()

# Compute, visualise confidence

confidence_grid = MRP.compute_confidence(iterations=param_MRP_iter)
plt.imshow(confidence_grid)
plt.title("SD-SMRP Confidence per pixel")
plt.show()

# Compute correlation between MAE and confidence

x1 = mae_grid.flatten()
x2 = x1[np.logical_not(np.isnan(x1))]

y1 = confidence_grid.flatten()
y2 = y1[np.logical_not(np.isnan(x1))]

corr = np.corrcoef(x2,y2)
print("SD-SMRP Error-Confidence Correlation: " + str(corr))


                
# Run WP-MRP

MRP = WP_SMRP(hidden_grid,feature_grid,SVR())
MRP.train()
pred_grid_WP = MRP.run(param_MRP_iter)

# Compute MAE, visualise pixel error

mae, mae_grid = MRP.mean_absolute_error(grid,gridded=True)
print("MAE WP-SMRP: " + str(mae))
plt.imshow(mae_grid)
plt.title("WP-SMRP Absolute error per pixel")
plt.show()

# Compute, visualise confidence

confidence_grid = MRP.compute_confidence(iterations=param_MRP_iter)
plt.imshow(confidence_grid)
plt.title("WP-SMRP Confidence per pixel")
plt.show()

# Compute correlation between MAE and confidence

x1 = mae_grid.flatten()
x2 = x1[np.logical_not(np.isnan(x1))]

y1 = confidence_grid.flatten()
y2 = y1[np.logical_not(np.isnan(x1))]

corr = np.corrcoef(x2,y2)
print("WP-SMRP Error-Confidence Correlation: " + str(corr))
                
                
                
                
                
                