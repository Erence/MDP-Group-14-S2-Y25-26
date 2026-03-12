from .geometry import face_to_theta
from .collision import collides_pose, within_bounds
from .geometry import theta_to_bin

_LOW_BAND_MAX_RATIO = 1.0 / 3.0
_MID_BAND_MAX_RATIO = 2.0 / 3.0


def _normalize_face(face):
    if isinstance(face, str):
        return face.upper()
    mapping = {0: "N", 2: "E", 4: "S", 6: "W"}
    if isinstance(face, int) and face in mapping:
        return mapping[face]
    raise ValueError(f"Unsupported face direction: {face}")


def _vertical_bias_for_obstacle_y(oy: float, cfg):
    low = float(getattr(cfg, "capture_vertical_bias_low_m", 0.0))
    mid = float(getattr(cfg, "capture_vertical_bias_mid_m", 0.0))
    high = float(getattr(cfg, "capture_vertical_bias_high_m", 0.0))

    y_min = float(cfg.min_center_y())
    y_max = float(cfg.max_center_y())
    if y_max <= y_min:
        return mid, "mid", 0.5

    ratio = (float(oy) - y_min) / (y_max - y_min)
    ratio = min(1.0, max(0.0, ratio))
    if ratio <= _LOW_BAND_MAX_RATIO:
        return low, "low", ratio
    if ratio <= _MID_BAND_MAX_RATIO:
        return mid, "mid", ratio
    return high, "high", ratio


def generate_view_states(obstacle: dict, cfg, hx, hy):
    ox = float(obstacle["x_m"])
    oy = float(obstacle["y_m"])
    face = _normalize_face(obstacle.get("face_dir", obstacle.get("d")))
    standoff_face_m = float(getattr(cfg, "capture_face_standoff_m", 0.0))
    vertical_bias = 0.0
    vertical_bias_band = None
    vertical_bias_ratio = None
    if standoff_face_m > 0.0:
        sensor_forward = float(getattr(cfg, "sensor_forward_offset_m", 0.0))
        if sensor_forward <= 0.0:
            sensor_forward = cfg.robot_L / 2.0
        # Convert requested face standoff to center-to-obstacle-center offset.
        offset = (cfg.obs_size / 2.0) + sensor_forward + standoff_face_m
    else:
        offset = float(cfg.capture_offset_cells) * cfg.cell_size

    # Single capture pose: offset along obstacle face normal, heading opposite to face.
    if face == "N":
        heading = face_to_theta("S")
        state = (ox, oy + offset, heading)
    elif face == "S":
        heading = face_to_theta("N")
        state = (ox, oy - offset, heading)
    elif face == "E":
        vertical_bias, vertical_bias_band, vertical_bias_ratio = _vertical_bias_for_obstacle_y(oy, cfg)
        heading = face_to_theta("W")
        state = (ox + offset, oy + vertical_bias, heading)
    else:
        vertical_bias, vertical_bias_band, vertical_bias_ratio = _vertical_bias_for_obstacle_y(oy, cfg)
        heading = face_to_theta("E")
        state = (ox - offset, oy + vertical_bias, heading)

    obstacle["_vertical_bias_band"] = vertical_bias_band
    obstacle["_vertical_bias_applied_m"] = float(vertical_bias)
    obstacle["_vertical_bias_y_ratio"] = vertical_bias_ratio

    # Single candidate validation.
    obs_xy = [(o["x_m"], o["y_m"]) for o in obstacle["all_obstacles_m"]]
    x, y, th = state
    if not within_bounds(x, y, cfg):
        return []
    tb = theta_to_bin(th, cfg.n_theta)
    if collides_pose(x, y, tb, obs_xy, cfg.obs_size / 2.0, hx, hy):
        return []
    return [state]
