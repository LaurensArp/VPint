# VPInt
This repository contains the code required to run VPint (value propagation-based spatial interpolation), associated with our DAMI publication VPint: value propagation-based spatial interpolation ( https://doi.org/10.1007/s10618-022-00843-2 ). Although working code is available for spatio-temporal interpolation, we recommend spatial interpolation as its preferred use case. The spatio-temporal code is at this point outdated, and its support may get dropped entirely in the future.

There are currently two versions of VPint, both of which are governed by update rules inspired by Markov reward processes. In the ideal case WP-MRP is used, which requires a target image (with NaNs denoting missing values, or with a mask of missing values) and a feature image containing values of the same area adhering to a similar spatial structure. If no feature data is available, SD-MRP can be used, though this will tend to regress to initialisation values over distance.

## Documentation

Please refer to the VPint documentation page at https://vpint.readthedocs.io/en/latest/ for detailed documentation of the code.

## Installation
To install VPint, please download the repository, navigate to the root folder, 
and run
`python setup.py install`

Dependencies:
* Python            3.6.12
* Numpy             1.19.0

Optional depdendencies:
* Scikit-image      0.17.2 (used for computing the SSIM performance metric)

## Running VPint
A minimal example to run VPint (WP-MRP), assuming appropriate data is already loaded:
```
from VPint.WP_MRP import WP_SMRP

MRP = WP_SMRP(target_data,feature_data)
result_data = MRP.run()
```

To run SD-MRP instead:
```
from VPint.SD_MRP import SD_SMRP

MRP = SD_SMRP(target_data)
# 100 iterations of random search for the best gamma parameter setting
MRP.find_gamma(100) 
result_data = MRP.run()
```

To run WP-MRP with randomly generated data:
```
from VPint.WP_MRP import WP_SMRP
from VPint.utils.hide_spatial_data import hide_values_uniform
import numpy as np

feature_scaling_param = 50 # for generating data
feature_noise_param = 0.5 # for generating data

true_data = np.random.rand(50,50) # create random ground truth data
target_data = hide_values_uniform(true_data,0.8) # hide 80% of true values
feature_data = true_data.copy() # identical features
feature_data = feature_data * feature_scaling_param # features in a different range
feature_data = feature_data + np.random.rand(50,50) * feature_noise_param # Adding noise

MRP = WP_SMRP(target_data,feature_data)
result_data = MRP.run()
```

To run WP-MRP with a machine learning model instead (useful in cases where the correlation between features and targets is less pronounced, but they are still related):

```
from VPint.WP_MRP import WP_SMRP
from sklearn.linear_model import LinearRegression

# Use any sklearn model (or other objects with train() and run() functions)
model = LinearRegression() 
MRP = WP_SMRP(target_data,feature_data,model=model)
MRP.train() # Train ML model on available data
result_data = MRP.run(method='predict')
```


## Citing this work

To read our 2022 DAMI paper on VPint, please see: https://doi.org/10.1007/s10618-022-00843-2
When publishing work using VPint, please cite:

    @article{ArpEtAl22,
        author = "Arp, Laurens and Baratchi, Mitra and Hoos, Holger",
        title = "VPint: value propagation-based spatial interpolation",
        journal = "Data Mining and Knowledge Discovery",
        volume = "36",
        pages = "",
        publisher = "Springer",
        year = "2022",
        issn = "1573-756X",
        doi = "https://doi.org/10.1007/s10618-022-00843-2",
        url = "https://link.springer.com/article/10.1007/s10618-022-00843-2",
    }