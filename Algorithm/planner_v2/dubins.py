import math
import time

from .asymmetric_words import CCC_WORDS, CSC_WORDS, solve_asymmetric_word_set
from .collision import collides_pose, within_bounds
from .geometry import theta_to_bin, wrap_pi


_DUBINS_GEAR = ((1, 1, 1),)


def _integrate_forward(x: float, y: float, th: float, signed_ds: float, kappa: float):
    if abs(kappa) < 1e-9:
        return x + signed_ds * math.cos(th), y + signed_ds * math.sin(th), wrap_pi(th)
    nth = wrap_pi(th + kappa * signed_ds)
    nx = x + (math.sin(nth) - math.sin(th)) / kappa
    ny = y + (-math.cos(nth) + math.cos(th)) / kappa
    return nx, ny, nth


def _segment_radius(seg_type: str, r_left: float, r_right: float):
    if seg_type == "L":
        return r_left
    if seg_type == "R":
        return r_right
    return None


def _segment_kappa(seg_type: str, r_left: float, r_right: float):
    if seg_type == "L":
        return 1.0 / r_left
    if seg_type == "R":
        return -1.0 / r_right
    return 0.0


def shortest_dubins_path(
    start_pose,
    goal_pose,
    turn_radius: float | None = None,
    *,
    r_left: float | None = None,
    r_right: float | None = None,
    deadline=None,
    allow_ccc: bool = True,
    residual_tol: float = 1e-2,
    max_newton_iters: int = 30,
    seed_mode: str = "compact",
    seed_limit: int = 0,
    early_exit_residual: float = 0.0,
):
    if r_left is None or r_right is None:
        if turn_radius is None:
            return {"success": False, "reason": "invalid_turn_radius"}
        r_left = float(turn_radius)
        r_right = float(turn_radius)

    words = list(CSC_WORDS)
    if allow_ccc:
        words.extend(CCC_WORDS)
    solved = solve_asymmetric_word_set(
        start_pose,
        goal_pose,
        float(r_left),
        float(r_right),
        words,
        _DUBINS_GEAR,
        max_cusps=0,
        residual_tol=residual_tol,
        deadline=deadline,
        max_newton_iters=max_newton_iters,
        seed_mode=seed_mode,
        seed_limit=seed_limit,
        early_exit_residual=early_exit_residual,
    )
    if not solved["success"]:
        reason = solved.get("reason", "no_solution")
        if reason == "time_budget":
            return {"success": False, "reason": "time_budget"}
        if reason == "invalid_turn_radius":
            return {"success": False, "reason": "invalid_turn_radius"}
        return {"success": False, "reason": "no_dubins_solution"}

    return {
        "success": True,
        "segments": solved["segments"],
        "length": solved["total_length_m"],
        "word": solved.get("word", ""),
        "residual": solved.get("residual", 0.0),
    }


def sample_dubins_path(start_pose, segments, step_len: float, r_left: float, r_right: float):
    if step_len <= 0.0:
        raise ValueError("step_len must be positive")

    x, y, th = start_pose
    samples = []

    for seg in segments:
        seg_type = seg["type"]
        gear = 1 if int(seg.get("gear", 1)) >= 0 else -1
        seg_len = max(0.0, float(seg["length_m"]))
        if seg_len <= 0.0:
            continue
        steps = max(1, int(math.ceil(seg_len / step_len)))
        ds = seg_len / steps
        kappa = _segment_kappa(seg_type, r_left, r_right)
        for _ in range(steps):
            x, y, th = _integrate_forward(x, y, th, gear * ds, kappa)
            samples.append((x, y, th))

    return samples


def _sample_and_validate_dubins(
    start_pose,
    segments,
    step_len: float,
    r_left: float,
    r_right: float,
    obstacles_xy,
    cfg,
    hx,
    hy,
    *,
    deadline=None,
    max_samples: int = 0,
):
    if step_len <= 0.0:
        return {"success": False, "reason": "invalid_substep_len", "samples": []}

    x, y, th = start_pose
    samples = []
    obs_half = cfg.obs_size / 2.0
    sample_cap = max(0, int(max_samples))
    sample_count = 0

    for seg in segments:
        seg_type = seg["type"]
        gear = 1 if int(seg.get("gear", 1)) >= 0 else -1
        seg_len = max(0.0, float(seg["length_m"]))
        if seg_len <= 0.0:
            continue
        steps = max(1, int(math.ceil(seg_len / step_len)))
        if sample_cap > 0 and sample_count + steps > sample_cap:
            return {"success": False, "reason": "dubins_sampling_cap", "samples": []}
        ds = seg_len / steps
        kappa = _segment_kappa(seg_type, r_left, r_right)
        for _ in range(steps):
            if deadline is not None and (sample_count & 31) == 0 and time.monotonic() >= deadline:
                return {"success": False, "reason": "time_budget", "samples": []}
            x, y, th = _integrate_forward(x, y, th, gear * ds, kappa)
            sample_count += 1
            if not within_bounds(x, y, cfg):
                return {"success": False, "reason": "dubins_collision", "samples": []}
            tb = theta_to_bin(th, cfg.n_theta)
            if collides_pose(x, y, tb, obstacles_xy, obs_half, hx, hy):
                return {"success": False, "reason": "dubins_collision", "samples": []}
            samples.append((x, y, th))

    return {"success": True, "samples": samples}


def dubins_to_commands(segments, r_left: float, r_right: float):
    commands = []
    for seg in segments:
        seg_type = seg["type"]
        gear = 1 if int(seg.get("gear", 1)) >= 0 else -1
        length_m = max(0.0, float(seg["length_m"]))
        if length_m <= 0.0:
            continue

        if seg_type == "S":
            centimeters = int(round(length_m * 100.0))
            if centimeters < 1:
                continue
            commands.append(f"FW{centimeters:03d}" if gear > 0 else f"BW{centimeters:03d}")
            continue

        radius = _segment_radius(seg_type, r_left, r_right)
        if radius is None:
            continue
        deg = int(round(math.degrees(length_m / radius)))
        if deg < 1:
            continue
        if seg_type == "L":
            commands.append(f"FL{deg:03d}" if gear > 0 else f"BL{deg:03d}")
        elif seg_type == "R":
            commands.append(f"FR{deg:03d}" if gear > 0 else f"BR{deg:03d}")
    return commands


def dubins_command_steps(start_pose, segments, r_left: float, r_right: float):
    x, y, th = start_pose
    steps = []

    for seg in segments:
        seg_type = seg["type"]
        gear = 1 if int(seg.get("gear", 1)) >= 0 else -1
        seg_len = max(0.0, float(seg["length_m"]))
        if seg_len <= 0.0:
            continue

        if seg_type == "S":
            centimeters = int(round(seg_len * 100.0))
            if centimeters < 1:
                continue
            ds = (centimeters / 100.0) * gear
            x, y, th = _integrate_forward(x, y, th, ds, 0.0)
            cmd = f"FW{centimeters:03d}" if gear > 0 else f"BW{centimeters:03d}"
            steps.append((cmd, (x, y, th)))
            continue

        radius = _segment_radius(seg_type, r_left, r_right)
        if radius is None:
            continue
        degrees = int(round(math.degrees(seg_len / radius)))
        if degrees < 1:
            continue
        dtheta = math.radians(degrees)
        ds = dtheta * radius * gear
        x, y, th = _integrate_forward(x, y, th, ds, _segment_kappa(seg_type, r_left, r_right))
        if seg_type == "L":
            cmd = f"FL{degrees:03d}" if gear > 0 else f"BL{degrees:03d}"
        else:
            cmd = f"FR{degrees:03d}" if gear > 0 else f"BR{degrees:03d}"
        steps.append((cmd, (x, y, th)))

    return steps


def plan_dubins_segment(start_pose, goal_pose, obstacles_xy, cfg, hx, hy, deadline=None):
    if deadline is not None and time.monotonic() >= deadline:
        return {
            "success": False,
            "reason": "time_budget",
            "path": [],
            "commands": [],
            "cost": float("inf"),
            "expanded": 0,
            "planner": "dubins",
        }

    r_left = cfg.turn_radius("L")
    r_right = cfg.turn_radius("R")

    shortest = shortest_dubins_path(
        start_pose,
        goal_pose,
        r_left=r_left,
        r_right=r_right,
        deadline=deadline,
        allow_ccc=bool(getattr(cfg, "dubins_allow_ccc", False)),
        residual_tol=float(getattr(cfg, "asym_solver_residual_tol", 1e-2)),
        max_newton_iters=int(getattr(cfg, "asym_solver_max_newton_iters", 14)),
        seed_mode=str(getattr(cfg, "asym_solver_seed_mode", "compact")),
        seed_limit=int(getattr(cfg, "asym_solver_seed_limit", 18)),
        early_exit_residual=float(getattr(cfg, "asym_solver_early_exit_residual", 0.003)),
    )
    if not shortest["success"]:
        return {
            "success": False,
            "reason": shortest["reason"],
            "path": [],
            "commands": [],
            "cost": float("inf"),
            "expanded": 0,
            "planner": "dubins",
        }

    segments = shortest["segments"]
    if deadline is not None and time.monotonic() >= deadline:
        return {
            "success": False,
            "reason": "time_budget",
            "path": [],
            "commands": [],
            "cost": float("inf"),
            "expanded": 0,
            "planner": "dubins",
        }
    sampled = _sample_and_validate_dubins(
        start_pose,
        segments,
        cfg.substep_len,
        r_left,
        r_right,
        obstacles_xy,
        cfg,
        hx,
        hy,
        deadline=deadline,
        max_samples=int(getattr(cfg, "analytic_sampling_max_points", 6000)),
    )
    if not sampled.get("success"):
        return {
            "success": False,
            "reason": sampled.get("reason", "dubins_collision"),
            "path": [],
            "commands": [],
            "cost": float("inf"),
            "expanded": 0,
            "planner": "dubins",
        }
    samples = sampled["samples"]

    path = [start_pose]
    path.extend(samples)
    path[-1] = goal_pose

    command_steps = dubins_command_steps(start_pose, segments, r_left, r_right)
    if command_steps:
        command_steps[-1] = (command_steps[-1][0], goal_pose)

    return {
        "success": True,
        "reason": None,
        "path": path,
        "commands": dubins_to_commands(segments, r_left, r_right),
        "command_steps": command_steps,
        "cost": shortest["length"],
        "expanded": 0,
        "planner": "dubins",
        "actions": [],
        "word": shortest.get("word", ""),
    }
