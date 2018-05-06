import h5py
import os
import scipy.misc
import numpy as np


class HDF5Gen(object):
    def __init__(self, experiment_path, episode_name, boolgenh5):
        self._boolgenh5 = boolgenh5
        self._h5file = os.path.join(experiment_path, episode_name + '.h5')

        self._car_start = False
        self.record = 1
        if self._boolgenh5:
            with h5py.File(self._h5file) as h5:
                h5.create_dataset('rgb', shape=(0, 88, 200, 3), dtype='uint8', maxshape=(None, 88, 200, 3))
                h5.create_dataset('targets', shape=(0, 4), dtype='float32', maxshape=(None, 4))

    def new_trail(self):
        self._car_start = False

    def save_data(self, measurements, sensor_data, control):
        if measurements.player_measurements.forward_speed > 0.1:
            self._car_start = True

        if self._boolgenh5:
            if self._car_start:
                image = sensor_data['CameraRGB'].data
                image = image[115:510, :]
                image = scipy.misc.imresize(image, [88, 200])

                target = np.array([
                    control.steer, control.throttle, control.brake,
                    measurements.player_measurements.forward_speed
                ], dtype=np.float32)

                with h5py.File(self._h5file) as h5:
                    h5['rgb'].resize((self.record, 88, 200, 3))
                    h5['rgb'][-1, :] = image
                    h5['targets'].resize((self.record, 4))
                    h5['targets'][-1, :] = target

                self.record += 1
