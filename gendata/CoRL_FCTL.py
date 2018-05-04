# Copyright (c) 2017 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# CORL experiment set.

from __future__ import print_function

import os

from benchmark import Benchmark
from experiment import Experiment
from carla.sensor import Camera
from carla.settings import CarlaSettings

from metrics import compute_summary
import numpy.random as random
import datetime


class CoRL_FCTL(Benchmark):

    def get_all_statistics(self):

        summary = compute_summary(os.path.join(
            self._full_name, self._suffix_name), [3])

        return summary

    def plot_summary_train(self):

        self._plot_summary([1.0, 3.0, 6.0, 8.0])

    def plot_summary_test(self):

        self._plot_summary([4.0, 14.0])

    def _plot_summary(self, weathers):
        """
        We plot the summary of the testing for the set selected weathers.
        The test weathers are [4,14]

        """

        metrics_summary = compute_summary(os.path.join(
            self._full_name, self._suffix_name), [3])

        for metric, values in metrics_summary.items():

            print('Metric : ', metric)
            for weather, tasks in values.items():
                if weather in set(weathers):
                    print('  Weather: ', weather)
                    count = 0
                    for t in tasks:
                        print('    Task ', count, ' -> ', t)
                        count += 1

                    print('    AvG  -> ', float(sum(tasks)) / float(len(tasks)))

    def _calculate_time_out(self, distance):
        """
        Function to return the timeout ( in miliseconds) that is calculated based on distance to goal.
        This is the same timeout as used on the CoRL paper.
        """

        return ((distance / 1000.0) / 10.0) * 3600.0 + 10.0

    def _poses_town01(self):
        """
        Each matrix is a new task. We have all the four tasks

        """

        def _poses_straight():
            return [[36, 40], [39, 35], [110, 114], [7, 3], [0, 4],
                    [61, 59], [33, 87], [80, 76], [45, 49], [84, 34],
                    [78, 70], [48, 35], [99, 31], [104, 82], [83, 101],
                    [67, 77]]

        return [_poses_straight()]

    def _poses_town02(self):

        def _poses_straight():
            return [[62, 57], [80, 7], [64, 66], [78, 76], [61, 0],
                    [73, 68], [45, 49], [54, 63], [51, 46], [53, 46],
                    [57, 82], [1, 56], [70, 66], [44, 0], [52, 77],
                    [78, 53]]

        return [_poses_straight()]


    def _build_experiments(self):
        """
        Creates the whole set of experiment objects,
        The experiments created depend on the selected Town.
        """

        # We set the camera
        # This single RGB camera is used on every experiment

        camera = Camera('CameraRGB')
        camera.set(FOV=100)

        camera.set_image_size(800, 600)

        camera.set_position(2.0, 0.0, 1.4)
        camera.set_rotation(-15.0, 0, 0)

        if self._city_name == 'Town01':
            weathers = [5, 6]
            poses_tasks = self._poses_town01()
            vehicles_tasks = [20]
            pedestrians_tasks = [0]
        else:
            weathers = [8, 9]
            poses_tasks = self._poses_town02()
            vehicles_tasks = [15]
            pedestrians_tasks = [0]

        experiments_vector = []

        for weather in weathers:

            for iteration in range(len(poses_tasks)):
                poses = poses_tasks[iteration]
                vehicles = vehicles_tasks[iteration]
                pedestrians = pedestrians_tasks[iteration]

                conditions = CarlaSettings()
                conditions.set(
                    SynchronousMode=True,
                    SendNonPlayerAgentsInfo=True,
                    NumberOfVehicles=vehicles,
                    NumberOfPedestrians=pedestrians,
                    WeatherId=weather,
                    SeedVehicles=random.randint(100000000, 999999999),
                    SeedPedestrians=random.randint(100000000, 999999999)
                )
                # Add all the cameras that were set for this experiments

                conditions.add_sensor(camera)

                experiment = Experiment()
                experiment.set(
                    Conditions=conditions,
                    Poses=poses,
                    Id=iteration,
                    Repetitions=1
                )
                experiments_vector.append(experiment)

        return experiments_vector

    def _get_details(self):

        # Function to get automatic information from the experiment for writing purposes
        return datetime.datetime.now().strftime("%Y%m%d%H%M") + '_' + self._city_name

    def _get_pose_and_task(self, line_on_file):
        """
        Returns the pose and task this experiment is, based on the line it was
        on the log file.
        """
        # We assume that the number of poses is constant
        return int(line_on_file / len(self._experiments)), line_on_file % 25
