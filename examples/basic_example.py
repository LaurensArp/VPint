import numpy as np
from sklearn.svm import SVR

from MRPinterpolation.SD_MRP import SD_MRP
from MRPinterpolation.WP_MRP import WP_MRP

# Create simple grid

grid = np.zeros((5,5))
for i in range(0,len(grid)):
    for j in range(0,len(grid[i])):
        if(np.random.rand() < 0.5):
            grid[i][j] = np.nan
        else:
            grid[i][j] = np.random.rand()
            
# Run SD-MRP

MRP = SD_MRP(grid)
MRP.find_gamma(100,0.5)
pred_grid_SD = MRP.run(100)

# Create random feature grid

f_grid = np.zeros((5,5,1))
for i in range(0,len(grid)):
    for j in range(0,len(grid[i])):
        f_grid[i][j][0] = np.random.rand()
        
# Run WP-MRP

MRP = WP_MRP(grid,f_grid,SVR())
MRP.train()
pred_grid_WP = MRP.run(100)

# Print results

print(grid)
print(pred_grid_SD)
print(pred_grid_WP)