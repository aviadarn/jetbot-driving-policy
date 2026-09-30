"""The auto-labeller: where a careful human would have clicked.

In 2024 each frame was labelled by clicking the point on the floor the car should steer
toward. Here that point is the centreline `lookahead` metres past the car's closest point,
projected through the same fisheye model the renderer uses.
"""
import numpy as np

from .camera import Fisheye, world_to_cam
from .config import TRACK

_FISH = None


def fisheye():
    global _FISH
    if _FISH is None:
        _FISH = Fisheye()
    return _FISH


def click(track, pose, hint=None, lookahead=TRACK.lookahead):
    """Pixel (u, v) in the 224 frame for a car pose, plus the track index used."""
    s, _, i = track.locate(pose[0], pose[1], hint=hint)
    j = track.index(s + lookahead)
    target = np.array([track.xy[j, 0], track.xy[j, 1], 0.0])
    u, v = fisheye().project(world_to_cam(target, pose))
    return float(u), float(v), i


def bucket(track, s, lookahead=TRACK.lookahead, straight_kappa=0.08):
    """straight / left / right by mean curvature over the lookahead window (not by label x,
    which confuses 'off-centre on a straight' with 'in a turn')."""
    k = track.mean_curvature_ahead(s, lookahead + 0.5)
    if abs(k) < straight_kappa:
        return "straight"
    return "left" if k > 0 else "right"


class ExpertPolicy:
    """The labeller used as a driver: perfect clicks, encoded in a label convention."""
    needs_image = False

    def __init__(self, track, mode="symmetric"):
        from .labels import encode
        self.track, self.mode, self._encode, self.hint = track, mode, encode, None

    def __call__(self, img, pose):
        u, v, self.hint = click(self.track, pose, hint=self.hint)
        return self._encode(u, v, self.mode)
