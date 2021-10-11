This repository contains the basic code required to run value propagation MRP-based spatial (SMRP) and spatio-temporal (STMRP) interpolation tasks. 
There are currently two MRP-based interpolation variants for both
tasks: SD-MRP and WP-MRP, which can be either spatial (SMRP) or 
spatio-temporal (STMRP). 
Further functionality will be added as the need for it arises.

Dependencies:
- Python            3.6.12
- Numpy             1.19.0

Optional depdendencies:
- Scikit-image      0.17.2

For reproducibility purposes, see "legacy" code. This implementation
used a functionally equivalent graph-based implementation of our methods, 
which was inefficient in terms of running time. The legacy code also
has a large amount of dependencies (see the README), and
was used to obtain results for the GDP and Covid datasets. 

To view the initial Master thesis associated with this code, please
see: https://ada.liacs.nl/papers/ArpEtAl20b.pdf 

When using this code for publications, please cite:

    @mastersthesis{Arp2020Markov,
        Author = {Arp, Laurens},
        Title = {A Markov Reward Process-Based Approach to Spatial Interpolation},
        Booktitle = {Master Thesis Computer Science},
        Year = {2020}
    }