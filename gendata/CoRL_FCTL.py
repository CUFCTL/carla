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
                    [78, 70]]

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

        weathers = [5, 6, 8, 9]

        poses_tasks = self._poses_town01()
        vehicles_tasks = [20]
        pedestrians_tasks = [50]

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
                    SeedVehicles=123456789,
                    SeedPedestrians=123456789
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
        return 'corl2017_' + self._city_name

    def _get_pose_and_task(self, line_on_file):
        """
        Returns the pose and task this experiment is, based on the line it was
        on the log file.
        """
        # We assume that the number of poses is constant
        return int(line_on_file / len(self._experiments)), line_on_file % 25
