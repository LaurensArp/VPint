This repository contains the basic code required to run value propagation MRP-based spatial (SMRP) and spatio-temporal (STMRP) interpolation tasks. 
There are currently two MRP-based interpolation variants for both
tasks: SD-MRP and WP-MRP, which can be either spatial (SMRP) or 
spatio-temporal (STMRP). 
Further functionality will be added as the need for it arises.

Dependencies:
- Python            3.6.12
- Numpy             1.19.0
- Networkx          2.4

For reproducibility purposes, see "legacy" code. This implementation
used the same graph-based methods as the "old" run functions in the
main code. It has a large amount of dependencies (see the README), and
was used to obtain results for the GDP and Covid datasets. Functionally,
it should be the same as using the "old" run functions, albeit more 
messy and less efficient.