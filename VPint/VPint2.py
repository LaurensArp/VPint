"""Wrapper for VPint code; most of the VPint2 functionality is already contained in the core code.
This module aims to improve the ease of use of VPint2 for EO tasks.
"""

import numpy as np

import multiprocessing


from .WP_MRP import *

class VPint2_interpolator():
    def __init__(self, target, features, mask=None, clip_target=None, clip_features=None, dtype=None, **kwargs):
        assert target.shape == features.shape
        assert len(target.shape) == 3
        assert len(features.shape) == 3

        if(dtype is not None):
            self.DTYPE = dtype
        else:
            self.DTYPE = np.float32

        # clip_target should be int, float (only max) or array-like with (min, max) values
        if(clip_target is not None):
            if(type(clip_target) == type(2) or type(clip_target) == type(2.1)):
                target = np.clip(target, a_min=-np.inf, a_max=clip_target)
            else:
                target = np.clip(target, a_min=clip_target[0], a_max=clip_target[1])

        if(clip_features is not None):
            if(type(clip_features) == type(2) or type(clip_features) == type(2.1)):
                features = np.clip(features, a_min=-np.inf, a_max=clip_features)
            else:
                features = np.clip(features, a_min=clip_features[0], a_max=clip_features[1])

        # Apply a cloud mask
        if(mask is not None):
            assert len(mask.shape) == 2
            assert mask.shape[0] == target.shape[0]
            assert mask.shape[1] == target.shape[1]
            self.target = self.apply_mask(target, mask, **kwargs).astype(self.DTYPE)
        else:
            self.target = target.astype(self.DTYPE)

        self.features = features.astype(self.DTYPE)


    def run(self, **kwargs):
        """Run VPint2 with parallelisation"""

        # Multiprocessing VPint version code written by Dean van Laar, slightly adapted
        
        manager = multiprocessing.Manager()
        pred_dict = manager.dict()
        pred = self.target.copy()
        grid_combos = []
        bands = []
        procs = len(range(0, self.target.shape[2]))

        # Create lists containing the bands in order
        for b in range(0, self.target.shape[2]):
            targetc = self.target[:, :, b]
            feature = self.features[:, :, b]
            band0 = b
            grid_combos.append([targetc, feature])
            bands.append(band0)

        # Start the processes
        jobs = []
        for i in range(0, procs):
            process = multiprocessing.Process(target=self.VPint2_single,
                                            args=(pred_dict, grid_combos[i], bands[i]), kwargs=kwargs)
            jobs.append(process)
        for j in jobs:
            j.start()

        for j in jobs:
            j.join()

            # Ensure all of the processes have finished
            if j.is_alive():
                pass
                #print("Job is not finished!")

        #Sort the dictionary after running VPint on the bands
        sorted_dict = dict(sorted(pred_dict.items()))
        pred0 = np.array([*sorted_dict.values()])
        pred1 = pred0.swapaxes(0, 2)
        pred_final = pred1.swapaxes(0, 1)

        return(pred_final)
    

    def VPint2_single(self, pred_dict, grids, band, **kwargs):
        interp = WP_SMRP(grids[0], grids[1])
        pred_dict[band] = interp.run(**kwargs)


    def run_serial(self, **kwargs):
        """Run VPint2 without parallelisation"""

        pred = self.target.copy()

        for b in range(0, pred.shape[2]):
            interp = WP_SMRP(self.target[:,:,b], self.features[:,:,b]) 
            pred[:,:,b] = interp.run(**kwargs)


        return(pred)


    def apply_mask(self, target, mask, threshold=0.2):
        """Automatically apply a cloud mask to convert arrays with missing values as NaN. Inefficient loop implementation for now."""
        target_cloudy = target.copy()

        for i in range(0, target_cloudy.shape[0]):
            for j in range(0, target_cloudy.shape[1]):
                if(mask[i,j] > threshold):
                    a = np.ones(target_cloudy.shape[2]) * np.nan
                    target_cloudy[i,j,:] = a

        return(target_cloudy)