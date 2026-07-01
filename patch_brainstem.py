import numpy as np
from scipy.ndimage import center_of_mass
from monai.transforms import MapTransform

class BrainstemPatchd(MapTransform):
    def __init__(self, keys, label_key="brainstem", patch_size=(64, 64, 64), vertebrae=None, label_key_original=None):
        super().__init__(keys)
        self.label_key = label_key
        self.patch_size = patch_size
        self.vertebrae = vertebrae
        self.label_key_original = label_key_original

    def __call__(self, data):
        d = dict(data)

        if self.label_key_original is not None:
            brainstem = d[self.label_key_original][0] > 0
        else:
            brainstem = d[self.label_key][0] > 0
        com = [int(round(c)) for c in center_of_mass(brainstem)]

        X, Y, Z = brainstem.shape
        sx, sy, sz = self.patch_size

        z_start = self.vertebrae[d['key']]
        z_end = z_start + sz
        if z_end > brainstem.shape[2]:
            z_end = brainstem.shape[2]
            z_start = z_end - sz

        x1 = max(0, com[0] - sx // 2)
        y1 = max(0, com[1] - sy // 2)

        x2 = x1 + sx
        y2 = y1 + sy

        if x2 > X:
            x2 = X
            x1 = X - sx
        if y2 > Y:
            y2 = Y
            y1 = Y - sy

        for key in self.keys:
            d[key] = d[key][:, x1:x2, y1:y2, z_start:z_end]
        if d[self.label_key].shape != (1, *self.patch_size):
            raise ValueError(f"Extracted patch shape {d['brainstem'].shape} does not match expected shape {(1, *self.patch_size)} for {d['key']}.")

        return d