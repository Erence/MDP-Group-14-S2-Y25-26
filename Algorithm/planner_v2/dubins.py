import math

from .collision import collides_swept
from .geometry import wrap_pi


def _mod2pi(angle: float) -> float:
    return angle % (2.0 * math.pi)


def _dubins_lsl(alpha: float, beta: float, d: float):
    tmp0 = d + math.sin(alpha) - math.sin(beta)
    p_sq = 2.0 + d * d - 2.0 * math.cos(alpha - beta) + 2.0 * d * (math.sin(alpha) - math.sin(beta))
    if p_sq < 0.0:
        return None
    tmp1 = math.atan2(math.cos(beta) - math.cos(alpha), tmp0)
    t = _mod2pi(-alpha + tmp1)
    p = math.sqrt(max(0.0, p_sq))
    q = _mod2pi(beta - tmp1)
    return ("LSL", (t, p, q))


def _dubins_rsr(alpha: float, beta: float, d: float):
    tmp0 = d - math.sin(alpha) + math.sin(beta)
    p_sq = 2.0 + d * d - 2.0 * math.cos(alpha - beta) + 2.0 * d * (-math.sin(alpha) + math.sin(beta))
    if p_sq < 0.0:
        return None
    tmp1 = math.atan2(math.cos(alpha) - math.cos(beta), tmp0)
    t = _mod2pi(alpha - tmp1)
    p = math.sqrt(max(0.0, p_sq))
    q = _mod2pi(-beta + tmp1)
    return ("RSR", (t, p, q))


def _dubins_lsr(alpha: float, beta: float, d: float):
    p_sq = -2.0 + d * d + 2.0 * math.cos(alpha - beta) + 2.0 * d * (math.sin(alpha) + math.sin(beta))
    if p_sq < 0.0:
        return None
    p = math.sqrt(max(0.0, p_sq))
    tmp2 = math.atan2(-math.cos(alpha) - math.cos(beta), d + math.sin(alpha) + math.sin(beta)) - math.atan2(-2.0, p)
    t = _mod2pi(-alpha + tmp2)
    q = _mod2pi(-beta + tmp2)
    return ("LSR", (t, p, q))


def _dubins_rsl(alpha: float, beta: float, d: float):
    p_sq = -2.0 + d * d + 2.0 * math.cos(alpha - beta) - 2.0 * d * (math.sin(alpha) + math.sin(beta))
    if p_sq < 0.0:
        return None
    p = math.sqrt(max(0.0, p_sq))
    tmp2 = math.atan2(math.cos(alpha) + math.cos(beta), d - math.sin(alpha) - math.sin(beta)) - math.atan2(2.0, p)
    t = _mod2pi(alpha - tmp2)
    q = _mod2pi(beta - tmp2)
    return ("RSL", (t, p, q))


def _dubins_rlr(alpha: float, beta: float, d: float):
    tmp0 = (6.0 - d * d + 2.0 * math.cos(alpha - beta) + 2.0 * d * (math.sin(alpha) - math.sin(beta))) / 8.0
    if abs(tmp0) > 1.0:
        return None
    p = _mod2pi(2.0 * math.pi - math.acos(tmp0))
    tmp1 = math.atan2(math.cos(alpha) - math.cos(beta), d - math.sin(alpha) + math.sin(beta))
    t = _mod2pi(alpha - tmp1 + p / 2.0)
    q = _mod2pi(alpha - beta - t + p)
    return ("RLR", (t, p, q))


def _dubins_lrl(alpha: float, beta: float, d: float):
    tmp0 = (6.0 - d * d + 2.0 * math.cos(alpha - beta) + 2.0 * d * (-math.sin(alpha) + math.sin(beta))) / 8.0
    if abs(tmp0) > 1.0:
        return None
    p = _mod2pi(2.0 * math.pi - math.acos(tmp0))
    tmp1 = math.atan2(math.cos(alpha) - math.cos(beta), d + math.sin(alpha) - math.sin(beta))
    t = _mod2pi(-alpha - tmp1 + p / 2.0)
    q = _mod2pi(_mod2pi(beta) - alpha - t + p)
    return ("LRL", (t, p, q))


def shortest_dubins_path(start_pose, goal_pose, turn_radius: float):
    if turn_radius <= 0.0:
        return {"success": False, "reason": "invalid_turn_radius"}

    sx, sy, sth = start_pose
    gx, gy, gth = goal_pose

    dx = gx - sx
    dy = gy - sy
    distance = math.hypot(dx, dy)
    d = distance / turn_radius

    theta = math.atan2(dy, dx)
    alpha = _mod2pi(sth - theta)
    beta = _mod2pi(gth - theta)

    generators = (
        _dubins_lsl,
        _dubins_rsr,
        _dubins_lsr,
        _dubins_rsl,
        _dubins_rlr,
        _dubins_lrl,
    )

    best = None
    for fn in generators:
        res = fn(alpha, beta, d)
        if res is None:
            continue
        mode, params = res
        length = turn_radius * (params[0] + params[1] + params[2])
        if best is None or length < best["length"]:
            best = {
                "mode": mode,
                "params": params,
                "length": length,
            }

    if best is None:
        return {"success": False, "reason": "no_dubins_solution"}

    return {"success": True, **best}


def _integrate_forward(x: float, y: float, th: float, ds: float, kappa: float):
    if abs(kappa) < 1e-9:
        return x + ds * math.cos(th), y + ds * math.sin(th), wrap_pi(th)
    nth = wrap_pi(th + kappa * ds)
    nx = x + (math.sin(nth) - math.sin(th)) / kappa
    ny = y + (-math.cos(nth) + math.cos(th)) / kappa
    return nx, ny, nth


def sample_dubins_path(start_pose, mode: str, params, turn_radius: float, step_len: float):
    if step_len <= 0.0:
        raise ValueError("step_len must be positive")

    x, y, th = start_pose
    samples = []

    for seg_type, seg_val in zip(mode, params):
        if seg_type == "S":
            seg_length = max(0.0, seg_val * turn_radius)
            if seg_length <= 0.0:
                continue
            steps = max(1, int(math.ceil(seg_length / step_len)))
            ds = seg_length / steps
            for _ in range(steps):
                x, y, th = _integrate_forward(x, y, th, ds, 0.0)
                samples.append((x, y, th))
            continue

        if seg_type == "L":
            seg_length = max(0.0, seg_val * turn_radius)
            if seg_length <= 0.0:
                continue
            steps = max(1, int(math.ceil(seg_length / step_len)))
            ds = seg_length / steps
            for _ in range(steps):
                x, y, th = _integrate_forward(x, y, th, ds, 1.0 / turn_radius)
                samples.append((x, y, th))
            continue

        if seg_type == "R":
            seg_length = max(0.0, seg_val * turn_radius)
            if seg_length <= 0.0:
                continue
            steps = max(1, int(math.ceil(seg_length / step_len)))
            ds = seg_length / steps
            for _ in range(steps):
                x, y, th = _integrate_forward(x, y, th, ds, -1.0 / turn_radius)
                samples.append((x, y, th))
            continue

        raise ValueError(f"Unsupported Dubins segment type: {seg_type}")

    return samples


def dubins_to_commands(mode: str, params, turn_radius: float):
    commands = []
    for seg_type, seg_val in zip(mode, params):
        if seg_type == "S":
            centimeters = int(round(seg_val * turn_radius * 100.0))
            if centimeters >= 1:
                commands.append(f"FW{centimeters:03d}")
            continue

        deg = int(round(math.degrees(seg_val)))
        if deg < 1:
            continue
        if seg_type == "L":
            commands.append(f"FL{deg:03d}")
        elif seg_type == "R":
            commands.append(f"FR{deg:03d}")
        else:
            raise ValueError(f"Unsupported Dubins segment type: {seg_type}")
    return commands


def dubins_command_steps(start_pose, mode: str, params, turn_radius: float):
    x, y, th = start_pose
    steps = []

    for seg_type, seg_val in zip(mode, params):
        if seg_type == "S":
            centimeters = int(round(seg_val * turn_radius * 100.0))
            if centimeters < 1:
                continue
            ds = centimeters / 100.0
            x, y, th = _integrate_forward(x, y, th, ds, 0.0)
            steps.append((f"FW{centimeters:03d}", (x, y, th)))
            continue

        degrees = int(round(math.degrees(seg_val)))
        if degrees < 1:
            continue

        dtheta = math.radians(degrees)
        kappa = 1.0 / turn_radius if seg_type == "L" else -1.0 / turn_radius
        ds = dtheta * turn_radius
        x, y, th = _integrate_forward(x, y, th, ds, kappa)

        if seg_type == "L":
            steps.append((f"FL{degrees:03d}", (x, y, th)))
        elif seg_type == "R":
            steps.append((f"FR{degrees:03d}", (x, y, th)))
        else:
            raise ValueError(f"Unsupported Dubins segment type: {seg_type}")

    return steps


def plan_dubins_segment(start_pose, goal_pose, obstacles_xy, cfg, hx, hy):
    shortest = shortest_dubins_path(start_pose, goal_pose, cfg.r_min)
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

    mode = shortest["mode"]
    params = shortest["params"]
    samples = sample_dubins_path(start_pose, mode, params, cfg.r_min, cfg.substep_len)

    if samples and collides_swept(samples, obstacles_xy, cfg, hx, hy):
        return {
            "success": False,
            "reason": "dubins_collision",
            "path": [],
            "commands": [],
            "cost": float("inf"),
            "expanded": 0,
            "planner": "dubins",
        }

    path = [start_pose]
    path.extend(samples)
    path[-1] = goal_pose

    command_steps = dubins_command_steps(start_pose, mode, params, cfg.r_min)
    if command_steps:
        # Anchor final command endpoint at goal pose for stable command-aligned UI stepping.
        command_steps[-1] = (command_steps[-1][0], goal_pose)

    return {
        "success": True,
        "reason": None,
        "path": path,
        "commands": dubins_to_commands(mode, params, cfg.r_min),
        "command_steps": command_steps,
        "cost": shortest["length"],
        "expanded": 0,
        "planner": "dubins",
        "actions": [],
        "mode": mode,
        "params": params,
    }
