from __future__ import print_function

import random
import os
import collections

from carla.client import CarlaClient
from carla import image_converter
from carla.sensor import Camera
from carla.settings import CarlaSettings
from carla.util import print_over_same_line
from PIL import Image as PImage


def print_measurements(measurements):
    number_of_agents = len(measurements.non_player_agents)
    player_measurements = measurements.player_measurements
    message = 'Vehicle at ({pos_x:.1f}, {pos_y:.1f}), '
    message += '{speed:.2f} km/h, '
    message += 'Collision: {{vehicles={col_cars:.0f}, pedestrians={col_ped:.0f}, other={col_other:.0f}}}, '
    message += '{other_lane:.0f}% other lane, {offroad:.0f}% off-road, '
    message += '({agents_num:d} non-player agents in the scene)'
    message = message.format(
        pos_x=player_measurements.transform.location.x / 100,  # cm -> m
        pos_y=player_measurements.transform.location.y / 100,
        speed=player_measurements.forward_speed,
        col_cars=player_measurements.collision_vehicles,
        col_ped=player_measurements.collision_pedestrians,
        col_other=player_measurements.collision_other,
        other_lane=100 * player_measurements.intersection_otherlane,
        offroad=100 * player_measurements.intersection_offroad,
        agents_num=number_of_agents)
    print_over_same_line(message)


class CarlaEnv(object):
    def __init__(self,
                 host='eceftl1.ces.clemson.edu',
                 port=2000,
                 image_filename_format='_images/episode_{:0>3d}/{:s}/image_{:0>5d}.png',
                 save_images_to_disk=False):
        self.image_filename_format = image_filename_format
        self.save_images_to_disk = save_images_to_disk

        self.host = host
        self.port = port
        self.timeout = 60

        self._Observation = collections.namedtuple('Observation', ['image', 'label'])
        self.frame = 0
        self.episode_index = -1

        self.client = CarlaClient(self.host, self.port, self.timeout)

    def connect(self):
        self.client.connect()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.client.disconnect()

    def start_new_episode(self, player_index):
        settings = CarlaSettings()
        settings.set(
            SynchronousMode=True,
            SendNonPlayerAgentsInfo=True,
            NumberOfVehicles=20,
            NumberOfPedestrians=40,
            WeatherId=random.choice([1, 3, 7, 8, 14]))
        settings.randomize_seeds()

        camera0 = Camera('CameraRGB')
        camera0.set(CameraFOV=100)
        camera0.set_image_size(800, 400)
        camera0.set_position(120, 0, 130)
        settings.add_sensor(camera0)

        camera2 = Camera('CameraSeg', PostProcessing='SemanticSegmentation')
        camera2.set(CameraFOV=100)
        camera2.set_image_size(800, 400)
        camera2.set_position(120, 0, 130)
        settings.add_sensor(camera2)

        self.client.load_settings(settings)
        self.episode_index += 1
        self.frame = 0
        print('\nStarting episode {0}.'.format(self.episode_index))
        self.client.start_episode(player_index)

    def _get_observation(self, sensor_data):
        camera_rgb_data = sensor_data['CameraRGB']
        camera_rgb_img = PImage.frombytes(
            mode='RGBA',
            size=(camera_rgb_data.width, camera_rgb_data.height),
            data=camera_rgb_data.raw_data,
            decoder_name='raw')
        b, g, r, a = camera_rgb_img.split()
        camera_rgb_img = PImage.merge("RGB", (r, g, b))

        camera_seg_data = sensor_data['CameraSeg']
        camera_seg_gray = image_converter.labels_to_grayscale2(camera_seg_data).astype('uint8')

        camera_seg_img = PImage.fromarray(camera_seg_gray, 'L')

        return self._Observation(camera_rgb_img, camera_seg_img)

    def _gen_images(self, observation):
        camera_rgb_img_filename = self.image_filename_format.format(self.episode_index, 'CameraRGB', self.frame)
        camera_seg_img_filename = self.image_filename_format.format(self.episode_index, 'CameraSeg', self.frame)

        rgb_folder = os.path.dirname(camera_rgb_img_filename)
        if not os.path.isdir(rgb_folder):
            os.makedirs(rgb_folder)

        seg_folder = os.path.dirname(camera_seg_img_filename)
        if not os.path.isdir(seg_folder):
            os.makedirs(seg_folder)

        observation.image.save(camera_rgb_img_filename)
        observation.label.save(camera_seg_img_filename)

    def auto_drive(self):
        measurements, sensor_data = self.client.read_data()
        print_measurements(measurements)

        observation = self._get_observation(sensor_data)
        if self.save_images_to_disk:
            self._gen_images(self._get_observation(sensor_data))

        control = measurements.player_measurements.autopilot_control
        control.steer += random.uniform(-0.1, 0.1)
        self.client.send_control(control)
        self.frame += 1

    def step(self, *args, **kwargs):
        measurements, sensor_data = self.client.read_data()
        print_measurements(measurements)

        observation = self._get_observation(sensor_data)
        if self.save_images_to_disk:
            self._gen_images(observation)

        pm = measurements.player_measurements
        rewards = pm.collision_vehicles + pm.collision_pedestrians + pm.collision_other
        self.client.send_control(*args, **kwargs)
        self.frame += 1
        return observation, rewards
