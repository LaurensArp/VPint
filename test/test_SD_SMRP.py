import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'../'))

import pytest

import numpy as np
import networkx as nx

from VPint.SD_MRP import SD_SMRP

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
    
def test_init_SD_SMRP(init_grid):
    MRP = SD_SMRP(init_grid)
    assert (np.nansum(MRP.original_grid - init_grid) < 1e-6)
    assert (np.nansum(MRP.pred_grid - init_grid) < 1e-6)
    assert isinstance(MRP.gamma, float)
    
def test_run_interpolates_SD_SMRP(init_grid):
    MRP = SD_SMRP(init_grid)
    pred_grid = MRP.run(1)
    assert not(np.isnan(pred_grid).any())
    assert (np.sum(pred_grid - MRP.pred_grid) < 1e-6)
    
def test_set_gamma(init_grid):
    MRP = SD_SMRP(init_grid)
    MRP.set_gamma(0.5)
    assert MRP.gamma == 0.5
    MRP.set_gamma(0.9)
    assert MRP.gamma == 0.9
    
def test_run_shape_SD_SMRP(init_grid):
    MRP = SD_SMRP(init_grid)
    pred_grid = MRP.run(1)
    assert pred_grid.shape == init_grid.shape
    
def test_find_gamma(init_grid):
    MRP = SD_SMRP(init_grid)
    MRP.find_gamma(1,0.5)
    assert isinstance(MRP.gamma, float)