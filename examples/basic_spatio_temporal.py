import numpy as np
from sklearn.svm import SVR

from MRPinterpolation.STMRP import SD_STMRP, WP_STMRP

height = 5
width = 5
depth = 5

# Create random grid

grid = np.zeros((height,width))
for i in range(0,height):
    for j in range(0,width):
        grid[i][j] = np.random.rand()
        
# Create random feature grid
        
f_grid = np.zeros((height,width,1))
for i in range(0,height):
    for j in range(0,width):
        f_grid[i][j][0] = np.random.rand()
            

# Create temporal layers with different hidden cells
            
data = {}
counter = 2020
stamp = str(counter) + "-02-10 14:56:01"

for t in range(0,depth):
    new_grid = grid.copy()
    for i in range(0,height):
        for j in range(0,width):
            if(np.random.rand() < 0.5):
                new_grid[i][j] = np.nan
    data[stamp] = new_grid
    counter += np.random.randint(low=1,high=3)
    stamp = stamp = str(counter) + "-02-10 14:56:01"
    
# SD-STMRP
    
print("SD-STMRP")
    
MRP = SD_STMRP(data,auto_timestamps=True)
gamma, tau = MRP.find_discounts(100,0.5)
pred_grid = MRP.run(100)

print(pred_grid)

# WP-STMRP

print("WP-STMRP")

MRP = WP_STMRP(data,f_grid,SVR(),auto_timestamps=True)
MRP.train()
pred_grid = MRP.run(100)

print(pred_grid)