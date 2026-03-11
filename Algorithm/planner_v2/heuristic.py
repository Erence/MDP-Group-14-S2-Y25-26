import math
from .geometry import wrap_pi


def heuristic(x: float, y: float, th: float, gx: float, gy: float, gth: float, cfg) -> float:
    dist = math.hypot(gx - x, gy - y)
    dth = abs(wrap_pi(gth - th))

    # Lower bound for curvature-constrained travel (very lightweight approximation)
    turning_lb = cfg.min_turn_radius() * dth
    return max(dist, turning_lb) + cfg.w_heading * dth
