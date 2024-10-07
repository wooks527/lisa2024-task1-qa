from monai.transforms import Transform, RandRotated
import numpy as np


class RandRotateByAxisd(Transform):
    def __init__(self, keys, degrees=15, prob=0.5, keep_size=True):
        self.keys = keys
        self.degrees = degrees
        self.prob = prob
        self.keep_size = keep_size

    def __call__(self, data):
        if self.degrees == 0:
            return data

        # 2 == sag == x
        if (
            isinstance(data["label"], np.ndarray)
            and data["label"][-1] == 2
            or isinstance(data["label"], np.int64)
            and data["label"] == 2
        ):
            range_x = np.deg2rad(self.degrees)
            range_y = 0
            range_z = 0
        # 1 == cor == y
        elif (
            isinstance(data["label"], np.ndarray)
            and data["label"][-1] == 1
            or isinstance(data["label"], np.int64)
            and data["label"] == 1
        ):
            range_x = 0
            range_y = np.deg2rad(self.degrees)
            range_z = 0
        # 0 == axi == z
        elif (
            isinstance(data["label"], np.ndarray)
            and data["label"][-1] == 0
            or isinstance(data["label"], np.int64)
            and data["label"] == 0
        ):
            range_x = 0
            range_y = 0
            range_z = np.deg2rad(self.degrees)
        else:
            assert False, "There is no axis information!"

        rotation_transform = RandRotated(
            keys=self.keys,
            range_x=range_x,
            range_y=range_y,
            range_z=range_z,
            prob=self.prob,
            keep_size=self.keep_size,
        )
        return rotation_transform(data)
