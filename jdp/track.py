"""Seeded cone corridors: straights and left/right arcs, like the 2024 hallway course.

A track is an open corridor, not a loop, so left and right turns can be mixed freely.
Curvature is piecewise constant, then smoothed over 0.5 m so the path has no kinks.
"""
import math
import numpy as np
from scipy.spatial import cKDTree

from .config import TRACK

SEG_P = (0.34, 0.33, 0.33)  # straight, left, right


class Track:
    def __init__(self, seed, length=TRACK.length, cfg=TRACK):
        self.seed = seed
        self.cfg = cfg
        rng = np.random.default_rng(seed)
        for _ in range(200):
            kappa = _curvature_profile(rng, length, cfg.ds)
            xy, heading = _integrate(kappa, cfg.ds)
            if not _self_intersects(xy, cfg):
                break
        else:
            raise RuntimeError(f"track seed {seed}: no non-intersecting layout in 200 tries")
        self.kappa = kappa
        self.xy = xy
        self.heading = heading
        self.s = np.arange(len(kappa)) * cfg.ds
        self.length = float(self.s[-1])
        self._tree = cKDTree(xy)

    def reversed(self):
        """Same corridor driven the other way (eval uses both directions)."""
        t = object.__new__(Track)
        t.seed, t.cfg = self.seed, self.cfg
        t.xy = self.xy[::-1].copy()
        t.heading = _wrap(self.heading[::-1] + math.pi)
        t.kappa = -self.kappa[::-1].copy()
        t.s = self.s.copy()
        t.length = self.length
        t._tree = cKDTree(t.xy)
        return t

    def pose_at(self, s, lateral=0.0, dheading=0.0):
        i = self.index(s)
        h = self.heading[i]
        nx, ny = -math.sin(h), math.cos(h)  # left normal
        x, y = self.xy[i]
        return np.array([x + lateral * nx, y + lateral * ny, _wrap(h + dheading)])

    def index(self, s):
        return int(np.clip(round(s / self.cfg.ds), 0, len(self.s) - 1))

    def locate(self, x, y, hint=None, window=4.0):
        """Return (s, lateral, index): lateral is signed, left of travel direction positive."""
        if hint is None:
            _, i = self._tree.query((x, y))
        else:
            n = int(window / self.cfg.ds)
            lo, hi = max(0, hint - n), min(len(self.s), hint + n + 1)
            d2 = (self.xy[lo:hi, 0] - x) ** 2 + (self.xy[lo:hi, 1] - y) ** 2
            i = lo + int(np.argmin(d2))
        h = self.heading[i]
        dx, dy = x - self.xy[i, 0], y - self.xy[i, 1]
        lateral = -math.sin(h) * dx + math.cos(h) * dy
        return float(self.s[i]), float(lateral), int(i)

    def mean_curvature_ahead(self, s, dist):
        i0, i1 = self.index(s), self.index(s + dist)
        return float(self.kappa[i0:i1 + 1].mean())

    def edges(self):
        """Left and right cone lines as polylines."""
        n = np.stack([-np.sin(self.heading), np.cos(self.heading)], axis=1)
        w = self.cfg.half_width
        return self.xy + w * n, self.xy - w * n

    def cone_positions(self):
        out = []
        for edge in self.edges():
            seg = np.linalg.norm(np.diff(edge, axis=0), axis=1)
            arc = np.concatenate([[0.0], np.cumsum(seg)])
            for a in np.arange(0.0, arc[-1], self.cfg.cone_spacing):
                j = int(np.searchsorted(arc, a))
                out.append(edge[min(j, len(edge) - 1)])
        return np.array(out)


def _curvature_profile(rng, length, ds):
    kappa = []
    while len(kappa) * ds < length:
        kind = rng.choice(3, p=SEG_P)
        if kind == 0:
            seg_len = rng.uniform(1.5, 5.0)
            k = 0.0
        else:
            radius = rng.uniform(2.2, 5.0)
            angle = rng.uniform(35, 110) * math.pi / 180
            seg_len = radius * angle
            k = (1.0 if kind == 1 else -1.0) / radius
        kappa += [k] * max(1, int(seg_len / ds))
    kappa = np.array(kappa[: int(length / ds)])
    lead = int(1.5 / ds)  # every track starts with a 1.5 m straight
    kappa = np.concatenate([np.zeros(lead), kappa])
    w = int(0.5 / ds)
    return np.convolve(np.pad(kappa, w, mode="edge"), np.ones(w) / w, mode="same")[w:-w]


def _integrate(kappa, ds):
    heading = np.concatenate([[0.0], np.cumsum(kappa[:-1] * ds)])
    xy = np.zeros((len(kappa), 2))
    xy[1:, 0] = np.cumsum(np.cos(heading[:-1]) * ds)
    xy[1:, 1] = np.cumsum(np.sin(heading[:-1]) * ds)
    return xy, _wrap(heading)


def _self_intersects(xy, cfg):
    """Corridor must keep its width away from any part of itself more than 4 m along the path."""
    tree = cKDTree(xy)
    clearance = 2 * cfg.half_width + 0.6
    gap = int(4.0 / cfg.ds)
    for i, j in tree.query_pairs(clearance):
        if abs(i - j) > gap:
            return True
    return False


def _wrap(a):
    return (np.asarray(a) + math.pi) % (2 * math.pi) - math.pi
