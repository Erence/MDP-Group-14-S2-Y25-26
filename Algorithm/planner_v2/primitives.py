from dataclasses import dataclass
import math
from .geometry import wrap_pi


@dataclass
class PrimitiveRollout:
    action: str
    next_gear: int
    cost_len: float
    end_pose: tuple
    samples: list


def _integrate_const_curvature(x: float, y: float, th: float, signed_ds: float, kappa: float):
    if abs(kappa) < 1e-9:
        nx = x + signed_ds * math.cos(th)
        ny = y + signed_ds * math.sin(th)
        nth = wrap_pi(th)
        return nx, ny, nth

    nth = wrap_pi(th + kappa * signed_ds)
    nx = x + (math.sin(nth) - math.sin(th)) / kappa
    ny = y + (-math.cos(nth) + math.cos(th)) / kappa
    return nx, ny, nth


def rollout_primitive(x: float, y: float, th: float, action: str, cfg):
    # action in {FS, FL, FR, RS, RL, RR}
    gear = 1 if action[0] == "F" else -1
    steer = action[1]

    if steer == "S":
        kappa = 0.0
    elif steer == "L":
        kappa = cfg.turn_curvature("L", gear)
    elif steer == "R":
        kappa = cfg.turn_curvature("R", gear)
    else:
        raise ValueError(f"Unknown steer in action: {action}")

    steps = max(1, int(round(cfg.primitive_len / cfg.substep_len)))
    ds = cfg.primitive_len / steps

    cx, cy, cth = x, y, th
    samples = []
    for _ in range(steps):
        signed_ds = gear * ds
        cx, cy, cth = _integrate_const_curvature(cx, cy, cth, signed_ds, kappa)
        samples.append((cx, cy, cth))

    return PrimitiveRollout(
        action=action,
        next_gear=gear,
        cost_len=cfg.primitive_len,
        end_pose=(cx, cy, cth),
        samples=samples,
    )


def enumerate_actions(current_gear: int, reverse_enabled: bool):
    actions = ["FS", "FL", "FR"]
    if reverse_enabled:
        actions.extend(["RS", "RL", "RR"])
    return actions
