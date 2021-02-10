import numpy as np
from sklearn.svm import SVR

from MRPinterpolation.SD_STMRP import SD_STMRP

height = 5
width = 5
depth = 5

grid = np.zeros((height,width))
for i in range(0,height):
    for j in range(0,width):
        grid[i][j] = np.random.rand()
            
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
    
MRP = SD_STMRP(data,auto_timestamps=True)
gamma, tau = MRP.find_discounts(100,0.5)
pred_grid = MRP.run(100)

print(pred_grid)