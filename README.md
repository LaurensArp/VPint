This repository contains the basic code required to run SD-MRP
or WP-MRP for 2D grid-based interpolation tasks. We intend to
add further functionality as new problems come up.

Dependencies:
- Numpy             1.19.0
- Networkx          2.4

Minimal usage example:

```python
import numpy as np
from sklearn.svm import SVR

from MRPinterpolation import SD_MRP, WP_MRP

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
```

```
[[0.22479289 0.95201257 0.70322903 0.49412834 0.86704105]
 [0.27206163 0.50847094        nan        nan 0.47276596]
 [       nan 0.65777289        nan 0.99338891 0.19563186]
 [0.74386984        nan 0.67423944 0.09532214        nan]
 [0.93720099        nan 0.46823068 0.94295567        nan]]
[[0.22479289 0.95201257 0.70322903 0.49412834 0.86704105]
 [0.27206163 0.50847094 0.16157796 0.08743723 0.47276596]
 [0.04837189 0.65777289 0.10925893 0.99338891 0.19563186]
 [0.74386984 0.18690648 0.67423944 0.09532214 0.05173091]
 [0.93720099 0.19986355 0.46823068 0.94295567 0.26527925]]
[[0.22479289 0.95201257 0.70322903 0.49412834 0.86704105]
 [0.27206163 0.50847094 0.24816966 0.1712835  0.47276596]
 [0.04262283 0.65777289 0.44381878 0.99338891 0.19563186]
 [0.74386984 0.47744721 0.67423944 0.09532214 0.17278127]
 [0.93720099 0.48667935 0.46823068 0.94295567 0.20951009]]
```