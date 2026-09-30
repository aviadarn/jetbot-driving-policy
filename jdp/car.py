"""Kinematic bicycle with the ride-on's two channels: throttle and a steering motor.

Steering command +1 means full right, matching right_motor in the 2024 code (a positive
image-x target produced a positive command). The steering actuator is unknown, so it is a
first-order lag toward clip(cmd, -1, 1) * max_steer. Commands can be delayed by
`latency_steps` frames to model inference time between capture and actuation.
"""
import math
from collections import deque
import numpy as np

from .config import CAR, DT


class Car:
    def __init__(self, pose, cfg=CAR, latency_steps=None):
        self.cfg = cfg
        self.x, self.y, self.psi = (float(v) for v in pose)
        self.delta = 0.0
        n = cfg.latency_steps if latency_steps is None else latency_steps
        self.queue = deque([(0.0, 0.0)] * n)

    @property
    def pose(self):
        return np.array([self.x, self.y, self.psi])

    def step(self, throttle, steer, dt=DT):
        self.queue.append((float(throttle), float(steer)))
        throttle, steer = self.queue.popleft()
        c = self.cfg
        target = -float(np.clip(steer, -1.0, 1.0)) * c.max_steer  # right = negative yaw
        self.delta += (target - self.delta) * (1.0 - math.exp(-dt / c.steer_tau))
        v = max(0.0, throttle) * c.speed_per_throttle
        self.psi += v / c.wheelbase * math.tan(self.delta) * dt
        self.psi = (self.psi + math.pi) % (2 * math.pi) - math.pi
        self.x += v * math.cos(self.psi) * dt
        self.y += v * math.sin(self.psi) * dt
        return v
