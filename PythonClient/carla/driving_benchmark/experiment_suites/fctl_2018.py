from __future__ import print_function

from carla.driving_benchmark.experiment import Experiment
from carla.sensor import Camera
from carla.settings import CarlaSettings
from carla.driving_benchmark.experiment_suites.experiment_suite import ExperimentSuite


class Fctl2018(ExperimentSuite):
    def __init__(self, city_name, times):
        self.times = times
        super(Fctl2018, self).__init__(city_name)


    @property
    def train_weathers(self):
        if self._city_name == 'Town01':
            return [5, 6]
        else:
            return [8, 9]

    @property
    def test_weathers(self):
        return []

    def _poses_town01(self):
        """
        Each matrix is a new task. We have all the four tasks

        """

        def _poses_straight():
            return [[36, 40], [39, 35], [110, 114], [7, 3], [0, 4],
                    [61, 59], [33, 87], [80, 76], [45, 49], [84, 34],
                    [78, 70]]

        def _poses_one_curve():
            return [[48, 35], [99, 31], [104, 82], [83, 101], [67, 77]]

        return [_poses_straight(), _poses_one_curve()] * self.times

    def _poses_town02(self):

        def _poses_straight():
            return [[62, 57], [80, 7], [64, 66], [78, 76], [61, 0],
                    [73, 68], [45, 49], [54, 63], [51, 46], [53, 46]]

        def _poses_one_curve():
            return [[57, 82], [1, 56], [70, 66], [44, 0], [52, 77], [78, 53]]

        return [_poses_straight(), _poses_one_curve()] * self.times

    def build_experiments(self):
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
            poses_tasks = self._poses_town01()
            vehicles_tasks = [0] * 10
            pedestrians_tasks = [0] * 10
        else:
            poses_tasks = self._poses_town02()
            vehicles_tasks = [0] * 10
            pedestrians_tasks = [0] * 10

        experiments_vector = []

        for weather in self.weathers:

            for iteration in range(len(poses_tasks)):
                poses = poses_tasks[iteration]
                vehicles = vehicles_tasks[iteration]
                pedestrians = pedestrians_tasks[iteration]

                conditions = CarlaSettings()
                conditions.set(
                    SendNonPlayerAgentsInfo=True,
                    NumberOfVehicles=vehicles,
                    NumberOfPedestrians=pedestrians,
                    WeatherId=weather
                )
                # Add all the cameras that were set for this experiments

                conditions.add_sensor(camera)

                experiment = Experiment()
                experiment.set(
                    Conditions=conditions,
                    Poses=poses,
                    Task=iteration,
                    Repetitions=1
                )
                experiments_vector.append(experiment)

        return experiments_vector
