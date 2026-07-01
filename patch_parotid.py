import numpy as np
from scipy.ndimage import center_of_mass
from monai.transforms import MapTransform

class ParotidPatchd(MapTransform):
    def __init__(self, keys, label_key="parotid", patch_size=(64, 64, 64), label_key_original=None):
        super().__init__(keys)
        self.label_key = label_key
        self.patch_size = patch_size
        self.label_key_original = label_key_original

    def __call__(self, data):
        d = dict(data)

        if self.label_key_original is not None:
            parotid = d[self.label_key_original][0] > 0
        else:
            parotid = d[self.label_key][0] > 0
        px, py, pz = self.patch_size

        coords = np.where(parotid)
        midx = (np.min(coords[0]) + np.max(coords[0])) // 2 + 1
        midy = (np.min(coords[1]) + np.max(coords[1])) // 2 + 1
        midz = (np.min(coords[2]) + np.max(coords[2])) // 2 + 1

        if midx-px//2 < 0:
            midx = px//2
        if midx+px//2 > parotid.shape[0]:
            midx = parotid.shape[0] - px//2
        if midy-py//2 < 0:
            midy = py//2
        if midy+py//2 > parotid.shape[1]:
            midy = parotid.shape[1] - py//2
        if midz-pz//2 < 0:
            midz = pz//2
        if midz+pz//2 > parotid.shape[2]:
            midz = parotid.shape[2] - pz//2

        for key in self.keys:
            d[key] = d[key][:, midx-px//2:midx+px//2, midy-py//2:midy+py//2, midz-pz//2:midz+pz//2]
        if d[self.label_key].shape != (1, *self.patch_size):
            raise ValueError(f"Extracted patch shape {d[self.label_key].shape} does not match expected shape {(1, *self.patch_size)} for {d['key']}.")

        return d