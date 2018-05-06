
from carla.agent.agent import Agent
import numpy as np


class AutoPilotAgent050(Agent):
    def run_step(self, measurements, sensor_data, directions, target):
        control = measurements.player_measurements.autopilot_control
        control.steer = control.steer + np.random.triangular(-0.5, 0, 0.5)
        if measurements.player_measurements.forward_speed > 30/3.6:
            control.throttle = 0.5

        return control
