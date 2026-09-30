"""The 2024 steering controller, ported line for line.

Source: rf_live_detection.ipynb, cell 13, in github.com/daisyKim12/autonomous_driving.
Statement order is kept exactly, including the speed being computed before the
deadband and before the 1.4 gain. Korean comments translated.

Outputs follow the car's wiring: `throttle` went to robot.left_motor and `steer`
to robot.right_motor (the ride-on's steering motor). Not differential drive.
"""
import math


class Controller2024:
    def __init__(self):
        self.angle = 0.0
        self.angle_last = 0.0

    def reset(self):
        self.angle = 0.0
        self.angle_last = 0.0

    def step(self, xy, stop=False):
        x = float(xy[0])
        y = (0.5 - float(xy[1])) / 2.0

        speed_offset = 0.34  # base speed

        angle_diff_bias = 0.1
        self.angle = math.atan2(x, y)  # steering angle
        if abs(self.angle) + angle_diff_bias < abs(self.angle_last):  # angle shrinking vs last frame
            angle_temp = -1 * self.angle * 0.3
        else:  # angle growing vs last frame
            angle_temp = self.angle
        speed_value = speed_offset - abs(angle_temp) * 0.05  # slow down while turning

        if abs(angle_temp) < 0.05:
            angle_temp = 0

        self.angle_last = self.angle

        angle_temp *= 1.4  # steering gain

        throttle = 0.0 if stop else speed_value
        return throttle, angle_temp
