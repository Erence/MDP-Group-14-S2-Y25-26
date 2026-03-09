from .geometry import face_to_theta
from .collision import collides_pose, within_bounds
from .geometry import theta_to_bin


def _normalize_face(face):
    if isinstance(face, str):
        return face.upper()
    mapping = {0: "N", 2: "E", 4: "S", 6: "W"}
    if isinstance(face, int) and face in mapping:
        return mapping[face]
    raise ValueError(f"Unsupported face direction: {face}")


def generate_view_states(obstacle: dict, cfg, hx, hy):
    ox = float(obstacle["x_m"])
    oy = float(obstacle["y_m"])
    face = _normalize_face(obstacle.get("face_dir", obstacle.get("d")))
    standoff_face_m = float(getattr(cfg, "capture_face_standoff_m", 0.0))
    vertical_bias = float(getattr(cfg, "capture_vertical_bias_m", 0.0))
    if standoff_face_m > 0.0:
        sensor_forward = float(getattr(cfg, "sensor_forward_offset_m", 0.0))
        if sensor_forward <= 0.0:
            sensor_forward = cfg.robot_L / 2.0
        # Convert requested face standoff to center-to-obstacle-center offset.
        offset = (cfg.obs_size / 2.0) + sensor_forward + standoff_face_m
    else:
        offset = float(cfg.capture_offset_cells) * cfg.cell_size

    # Single capture pose: offset along obstacle face normal, heading opposite to face.
    # Apply vertical bias to shift lateral aim point upward for N/S facing obstacles.
    if face == "N":
        heading = face_to_theta("S")
        state = (ox, oy + offset, heading)
    elif face == "S":
        heading = face_to_theta("N")
        state = (ox, oy - offset, heading)
    elif face == "E":
        heading = face_to_theta("W")
        state = (ox + offset, oy + vertical_bias, heading)
    else:
        heading = face_to_theta("E")
        state = (ox - offset, oy + vertical_bias, heading)

    # Single candidate validation.
    obs_xy = [(o["x_m"], o["y_m"]) for o in obstacle["all_obstacles_m"]]
    x, y, th = state
    if not within_bounds(x, y, cfg):
        return []
    tb = theta_to_bin(th, cfg.n_theta)
    if collides_pose(x, y, tb, obs_xy, cfg.obs_size / 2.0, hx, hy):
        return []
    return [state]
