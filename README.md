# VPInt
This repository contains the code required to run VPint (value propagation-based spatial interpolation), associated with our DAMI publication VPint: value propagation-based spatial interpolation ( https://doi.org/10.1007/s10618-022-00843-2 ). Although working code is available for spatio-temporal interpolation, we recommend spatial interpolation as its preferred use case. The spatio-temporal code is at this point outdated, and its support may get dropped entirely in the future.

There are currently two versions of VPint, both of which are governed by update rules inspired by Markov reward processes. In the ideal case WP-MRP is used, which requires a target image (with NaNs denoting missing values, or with a mask of missing values) and a feature image containing values of the same area adhering to a similar spatial structure. If no feature data is available, SD-MRP can be used, though this will tend to regress to initialisation values over distance.

## Documentation

The code contains docstraings with detailed explanations. We hope to move this documentation to ReadTheDocs when we can. 

## Installation
To install VPint (choose between requirements_slim.txt and requirements.txt based on desired level of optional functionalities):

```
conda create --name VPint python=3.12.3
conda activate VPint
git clone git@github.com:LaurensArp/VPint.git
cd VPint
pip install -r requirements.txt
python setup.py install
```

Optional depdendencies:
* Scikit-image      0.17.2 (used for computing the SSIM performance metric)
* Rasterio          1.3.10 (used by utils to load EO data products)

## Running VPint2 for cloud removal in Earth Observation data
A minimal example to run VPint2 cloud removal, assuming appropriate data is already loaded:

```
from VPint.VPint2 import VPint2_interpolator

VPint2 = VPint2_interpolator(target, features, mask=mask)
target_clean = VPint2.run()
```

For more details, please see the cloud_removal_example.ipynb notebook in the examples folder, and the accompanying blog post/tutorial (coming up).




## Citing this work

To read our 2024 ISPRS Journal of Photogrammetry and Remote Sensing paper on VPint2 for cloud removal, please see: https://doi.org/10.1016/j.isprsjprs.2024.07.030
When publishing work using VPint2 cloud removal, please cite:

```
@article{ArpEtAl24,
    author = {Laurens Arp and Holger Hoos and Peter {van Bodegom} and Alistair Francis and James Wheeler and Dean {van Laar} and Mitra Baratchi},
    title = {Training-free thick cloud removal for Sentinel-2 imagery using value propagation interpolation},
    journal = {ISPRS Journal of Photogrammetry and Remote Sensing},
    volume = {216},
    pages = {168-184},
    year = {2024},
    issn = {0924-2716},
    doi = "https://doi.org/10.1016/j.isprsjprs.2024.07.030",
    url = "https://www.sciencedirect.com/science/article/pii/S0924271624002995",
    tags = {Remote sensing, spatial interpolation, cloud removal},
}
```

To read our 2022 DAMI paper on VPint, please see: https://doi.org/10.1007/s10618-022-00843-2
When publishing work using VPint, please cite:

```
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
```