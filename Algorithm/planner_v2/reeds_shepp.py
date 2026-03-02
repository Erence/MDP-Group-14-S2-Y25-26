import math
import time

from .collision import collides_swept
from .geometry import wrap_pi
from .reeds_shepp_core import solve_reeds_shepp


def _integrate(x: float, y: float, th: float, signed_ds: float, kappa: float):
    if abs(kappa) < 1e-9:
        return x + signed_ds * math.cos(th), y + signed_ds * math.sin(th), wrap_pi(th)
    nth = wrap_pi(th + kappa * signed_ds)
    nx = x + (math.sin(nth) - math.sin(th)) / kappa
    ny = y + (-math.cos(nth) + math.cos(th)) / kappa
    return nx, ny, nth


def _segment_kappa(seg_type: str, turn_radius: float):
    if seg_type == "L":
        return 1.0 / turn_radius
    if seg_type == "R":
        return -1.0 / turn_radius
    return 0.0


def _sample_segments(start_pose, segments, step_len: float, turn_radius: float):
    if step_len <= 0.0:
        raise ValueError("step_len must be positive")

    x, y, th = start_pose
    samples = []
    for seg in segments:
        seg_len = max(0.0, float(seg["length_m"]))
        if seg_len <= 0.0:
            continue
        gear = 1.0 if int(seg["gear"]) >= 0 else -1.0
        steps = max(1, int(math.ceil(seg_len / step_len)))
        ds = seg_len / steps
        kappa = _segment_kappa(seg["type"], turn_radius)
        for _ in range(steps):
            x, y, th = _integrate(x, y, th, gear * ds, kappa)
            samples.append((x, y, th))
    return samples


def _segment_action(seg_type: str, gear: int):
    if seg_type == "S":
        return "FS" if gear > 0 else "RS"
    if seg_type == "L":
        return "FL" if gear > 0 else "RL"
    return "FR" if gear > 0 else "RR"


def _segment_opcode(seg_type: str, gear: int):
    if seg_type == "S":
        return "FW" if gear > 0 else "BW"
    if seg_type == "L":
        return "FL" if gear > 0 else "BL"
    return "FR" if gear > 0 else "BR"


def _commands_and_steps(start_pose, segments, turn_radius: float):
    x, y, th = start_pose
    commands = []
    steps = []

    for seg in segments:
        seg_type = seg["type"]
        gear = 1 if int(seg["gear"]) >= 0 else -1
        seg_len = max(0.0, float(seg["length_m"]))
        if seg_len <= 0.0:
            continue

        opcode = _segment_opcode(seg_type, gear)
        if seg_type == "S":
            units = int(round(seg_len * 100.0))
            if units < 1:
                continue
            ds = (units / 100.0) * gear
            x, y, th = _integrate(x, y, th, ds, 0.0)
            cmd = f"{opcode}{units:03d}"
        else:
            units = int(round(math.degrees(seg_len / turn_radius)))
            if units < 1:
                continue
            dtheta = math.radians(units)
            signed_ds = dtheta * turn_radius * gear
            x, y, th = _integrate(x, y, th, signed_ds, _segment_kappa(seg_type, turn_radius))
            cmd = f"{opcode}{units:03d}"

        commands.append(cmd)
        steps.append((cmd, (x, y, th)))

    return commands, steps


def plan_reeds_shepp_segment(start_pose, goal_pose, obstacles_xy, cfg, hx, hy, deadline=None):
    if not getattr(cfg, "rs_enabled", True):
        return {
            "success": False,
            "reason": "rs_disabled",
            "path": [],
            "commands": [],
            "command_steps": [],
            "cost": float("inf"),
            "expanded": 0,
            "planner": "reeds_shepp",
            "actions": [],
        }

    if not getattr(cfg, "reverse_enabled", True):
        return {
            "success": False,
            "reason": "reverse_disabled",
            "path": [],
            "commands": [],
            "command_steps": [],
            "cost": float("inf"),
            "expanded": 0,
            "planner": "reeds_shepp",
            "actions": [],
        }

    if deadline is not None and time.monotonic() >= deadline:
        return {
            "success": False,
            "reason": "time_budget",
            "path": [],
            "commands": [],
            "command_steps": [],
            "cost": float("inf"),
            "expanded": 0,
            "planner": "reeds_shepp",
            "actions": [],
        }

    solved = solve_reeds_shepp(
        start_pose,
        goal_pose,
        cfg.r_min,
        max_cusps=max(0, int(getattr(cfg, "rs_max_cusps", 2))),
        allow_ccc=bool(getattr(cfg, "rs_allow_ccc", True)),
        deadline=deadline,
    )
    if not solved["success"]:
        return {
            "success": False,
            "reason": solved.get("reason", "no_reeds_shepp_solution"),
            "path": [],
            "commands": [],
            "command_steps": [],
            "cost": float("inf"),
            "expanded": 0,
            "planner": "reeds_shepp",
            "actions": [],
        }

    segments = solved["segments"]
    samples = _sample_segments(start_pose, segments, cfg.substep_len, cfg.r_min)
    if samples and collides_swept(samples, obstacles_xy, cfg, hx, hy):
        return {
            "success": False,
            "reason": "reeds_shepp_collision",
            "path": [],
            "commands": [],
            "command_steps": [],
            "cost": float("inf"),
            "expanded": 0,
            "planner": "reeds_shepp",
            "actions": [],
        }

    path = [start_pose]
    path.extend(samples)
    if path:
        path[-1] = goal_pose

    commands, command_steps = _commands_and_steps(start_pose, segments, cfg.r_min)
    if command_steps:
        command_steps[-1] = (command_steps[-1][0], goal_pose)

    return {
        "success": True,
        "reason": None,
        "path": path,
        "commands": commands,
        "command_steps": command_steps,
        "cost": solved["total_length_m"],
        "expanded": 0,
        "planner": "reeds_shepp",
        "actions": [_segment_action(seg["type"], int(seg["gear"])) for seg in segments],
        "segments": segments,
        "word": solved.get("word", ""),
        "gear_signature": solved.get("gear_signature", ""),
        "cusp_count": solved.get("cusp_count", 0),
    }
