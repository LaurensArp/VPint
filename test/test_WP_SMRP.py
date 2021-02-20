import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'../'))

import numpy as np
import networkx as nx

import pytest

from sklearn.svm import SVR
from sklearn.utils.validation import check_is_fitted

from MRPinterpolation.WP_MRP import WP_SMRP

@pytest.fixture
def init_grid():
    grid = np.zeros((5,5))
    for i in range(0,grid.shape[0]):
        for j in range(0,grid.shape[1]):
            if(np.random.rand() < 0.5):
                grid[i][j] = np.nan
            else:
                grid[i][j] = np.random.rand()
    return(grid)
    
@pytest.fixture
def init_f_grid():
    f_grid = np.zeros((5,5,1))
    for i in range(0,f_grid.shape[0]):
        for j in range(0,f_grid.shape[1]):
            f_grid[i][j][0] = np.random.rand()
    return(f_grid)

def test_init_WP_SMRP(init_grid,init_f_grid):
    MRP = WP_SMRP(init_grid,init_f_grid,SVR())
    assert (np.nansum(MRP.original_grid - init_grid) < 1e-6)
    assert (np.nansum(MRP.pred_grid - init_grid) < 1e-6)
    assert (np.nansum(MRP.feature_grid - init_f_grid) < 1e-6)
    
def test_run_interpolates_WP_SMRP(init_grid,init_f_grid):
    MRP = WP_SMRP(init_grid,init_f_grid,SVR())
    MRP.train()
    pred_grid = MRP.run(1)
    assert not(np.isnan(pred_grid).any())
    assert (np.sum(pred_grid - MRP.pred_grid) < 1e-6)
    
def test_run_shape_WP_SMRP(init_grid,init_f_grid):
    MRP = WP_SMRP(init_grid,init_f_grid,SVR())
    MRP.train()
    pred_grid = MRP.run(1)
    assert pred_grid.shape == init_grid.shape
    
def test_fitted_WP_SMRP(init_grid,init_f_grid):
    MRP = WP_SMRP(init_grid,init_f_grid,SVR())
    MRP.train()
    check_is_fitted(MRP.model)
    assert True