import math
import time

from .collision import collides_pose, collides_swept, within_bounds
from .dubins import plan_dubins_segment
from .geometry import theta_to_bin


def _extra_scan_values(cfg):
    step = max(0.005, float(getattr(cfg, "dock_assist_scan_step_m", 0.02)))
    lo = max(0.0, float(getattr(cfg, "dock_assist_scan_min_extra_m", 0.02)))
    hi = max(lo, float(getattr(cfg, "dock_assist_scan_max_extra_m", 0.60)))

    vals = []
    count = max(0, int(math.floor((hi - lo) / step)))
    for i in range(count + 1):
        v = lo + i * step
        if v > hi + 1e-9:
            break
        vals.append(round(v, 6))
    if not vals:
        vals.append(round(lo, 6))
    if hi - vals[-1] > 1e-9:
        vals.append(round(hi, 6))
    return vals


def _straight_samples(pre_pose, to_pose, substep_len: float):
    px, py, pth = pre_pose
    tx, ty, _ = to_pose
    dist = math.hypot(tx - px, ty - py)
    if dist <= 1e-9:
        return []
    step = max(1e-3, float(substep_len))
    n = max(1, int(math.ceil(dist / step)))
    samples = []
    for i in range(1, n + 1):
        t = float(i) / float(n)
        x = px + (tx - px) * t
        y = py + (ty - py) * t
        samples.append((x, y, pth))
    return samples


def plan_dubins_dock_segment(from_pose, to_pose, obstacles_xy, cfg, hx, hy, deadline=None):
    if not bool(getattr(cfg, "dock_assist_enabled", True)):
        return {
            "success": False,
            "reason": "dubins_dock_disabled",
            "path": [],
            "actions": [],
            "commands": [],
            "command_steps": [],
            "cost": float("inf"),
            "expanded": 0,
            "planner": "dubins_dock",
        }

    obs_half = cfg.obs_size / 2.0
    tx, ty, tth = to_pose
    ux = math.cos(tth)
    uy = math.sin(tth)

    for extra_m in _extra_scan_values(cfg):
        if deadline is not None and time.monotonic() >= deadline:
            return {
                "success": False,
                "reason": "time_budget",
                "path": [],
                "actions": [],
                "commands": [],
                "command_steps": [],
                "cost": float("inf"),
                "expanded": 0,
                "planner": "dubins_dock",
            }

        px = tx - extra_m * ux
        py = ty - extra_m * uy
        pre_pose = (px, py, tth)

        if not within_bounds(px, py, cfg):
            continue
        tb = theta_to_bin(tth, cfg.n_theta)
        if collides_pose(px, py, tb, obstacles_xy, obs_half, hx, hy):
            continue

        straight_samples = _straight_samples(pre_pose, to_pose, cfg.substep_len)
        if straight_samples and collides_swept(straight_samples, obstacles_xy, cfg, hx, hy):
            continue

        dubins_seg = plan_dubins_segment(from_pose, pre_pose, obstacles_xy, cfg, hx, hy, deadline=deadline)
        if not dubins_seg.get("success"):
            if dubins_seg.get("reason") == "time_budget":
                return {
                    "success": False,
                    "reason": "time_budget",
                    "path": [],
                    "actions": [],
                    "commands": [],
                    "command_steps": [],
                    "cost": float("inf"),
                    "expanded": 0,
                    "planner": "dubins_dock",
                }
            continue

        forward_cm = max(1, int(round(extra_m * 100.0)))
        forward_m = float(forward_cm) / 100.0
        forward_cmd = f"FW{forward_cm:03d}"

        path = list(dubins_seg.get("path", []))
        if not path:
            path = [from_pose, pre_pose]
        if path:
            path[-1] = pre_pose
        if straight_samples:
            path.extend(straight_samples)
        if path:
            path[-1] = to_pose

        commands = list(dubins_seg.get("commands", []))
        commands.append(forward_cmd)

        command_steps = list(dubins_seg.get("command_steps", []))
        command_steps.append((forward_cmd, to_pose))

        return {
            "success": True,
            "reason": None,
            "path": path,
            "actions": [],
            "commands": commands,
            "command_steps": command_steps,
            "cost": float(dubins_seg.get("cost", 0.0)) + forward_m,
            "expanded": int(dubins_seg.get("expanded", 0)),
            "planner": "dubins_dock",
        }

    return {
        "success": False,
        "reason": "dubins_dock_no_path",
        "path": [],
        "actions": [],
        "commands": [],
        "command_steps": [],
        "cost": float("inf"),
        "expanded": 0,
        "planner": "dubins_dock",
    }
