import math
from .geometry import theta_to_bin


def within_bounds(x: float, y: float, cfg) -> bool:
    return cfg.min_center_x() <= x <= cfg.max_center_x() and cfg.min_center_y() <= y <= cfg.max_center_y()


def collides_pose(x: float, y: float, theta_bin: int, obstacles_xy, obs_half: float, hx, hy) -> bool:
    ex = obs_half + hx[theta_bin]
    ey = obs_half + hy[theta_bin]
    for ox, oy in obstacles_xy:
        if abs(x - ox) <= ex and abs(y - oy) <= ey:
            return True
    return False


def collides_swept(samples, obstacles_xy, cfg, hx, hy) -> bool:
    obs_half = cfg.obs_size / 2.0
    for x, y, th in samples:
        if not within_bounds(x, y, cfg):
            return True
        tb = theta_to_bin(th, cfg.n_theta)
        if collides_pose(x, y, tb, obstacles_xy, obs_half, hx, hy):
            return True
    return False


def min_obstacle_distance(x: float, y: float, obstacles_xy, obs_half: float) -> float:
    if not obstacles_xy:
        return 1e9
    best = 1e9
    for ox, oy in obstacles_xy:
        dx = max(0.0, abs(x - ox) - obs_half)
        dy = max(0.0, abs(y - oy) - obs_half)
        d = math.hypot(dx, dy)
        if d < best:
            best = d
    return best
