import numpy as np
import torch
from scipy.ndimage import center_of_mass
from monai.transforms import MapTransform

np.random.seed(42)  # For reproducibility of random margins

class SpinalCordPatchd(MapTransform):
    def __init__(self, keys, label_key="spinalcord", patch_size=(64, 64, 64), 
                 vertebrae=None, label_key_original=None):
        super().__init__(keys)
        self.label_key = label_key
        self.patch_size = patch_size
        self.vertebrae = vertebrae
        self.label_key_original = label_key_original

    def __call__(self, data):
        d = dict(data)

        mask = d[self.label_key][0] > 0  # [X, Y, Z]
        sx, sy, sz = self.patch_size

        z_start = self.vertebrae[d['key']]
        z_end = z_start + sz
        if z_end > mask.shape[2]:
            z_end = mask.shape[2]
            z_start = z_end - sz

        if self.label_key_original is not None:
            mask_original = d[self.label_key_original][0] > 0
            submask = mask_original[:, :, z_start:z_end]
            midx = (np.min(np.where(submask)[0]) + np.max(np.where(submask)[0])) // 2 + 1
            midy = (np.min(np.where(submask)[1]) + np.max(np.where(submask)[1])) // 2 + 1
        else:
            submask = mask[:, :, z_start:z_end]
            midx = (np.min(np.where(submask)[0]) + np.max(np.where(submask)[0])) // 2 + 1
            midy = (np.min(np.where(submask)[1]) + np.max(np.where(submask)[1])) // 2 + 1

        if midx-sx//2 < 0:
            midx = sx//2
        if midx+sx//2 > mask.shape[0]:
            midx = mask.shape[0] - sx//2
        if midy-sy//2 < 0:
            midy = sy//2
        if midy+sy//2 > mask.shape[1]:
            midy = mask.shape[1] - sy//2

        for key in self.keys:
            d[key] = d[key][:, midx-sx//2:midx+sx//2, midy-sy//2:midy+sy//2, z_start:z_end]
        if d[self.label_key].shape != (1, *self.patch_size):
            raise ValueError(f"Extracted patch shape {d[self.label_key].shape} does not match expected shape {(1, *self.patch_size)} for {d['key']}.")

        return d
