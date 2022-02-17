#VPInt
This repository contains the code required to run VPint (value propagation-based spatial interpolation). Although working code is available for spatio-temporal interpolation, we recommend spatial interpolation as its preferred use case. The spatio-temporal code is at this point outdated, and its support may get dropped entirely in the future.

There are currently two versions of VPint, both of which are governed by update rules inspired by Markov reward processes. In the ideal case WP-MRP is used, which requires a target image (with NaNs denoting missing values, or with a mask of missing values) and a feature image containing values of the same area adhering to a similar spatial structure. If no feature data is available, SD-MRP can be used, though this will tend to regress to initialisation values over distance.

##Installation
To install VPint, please download the repository, navigate to the root folder, 
and run
`python setup.py install`

Dependencies:
* Python            3.6.12
* Numpy             1.19.0

Optional depdendencies:
* Scikit-image      0.17.2 (used for computing the SSIM performance metric)

##Running VPint
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


##Citing this work

To view the initial Master thesis associated with this code, please
see: https://ada.liacs.nl/papers/ArpEtAl20b.pdf 

A paper for this work has been accepted for publication in the DAMI 
journal for the ECML-PKDD journal track (2022). A link to the paper, 
along with updated BibTeX information, will be added upon publication.
In the meantime, when using this code for publications, please cite:

    @mastersthesis{Arp2020Markov,
        Author = {Arp, Laurens},
        Title = {A Markov Reward Process-Based Approach to Spatial Interpolation},
        Booktitle = {Master Thesis Computer Science},
        Year = {2020}
    }