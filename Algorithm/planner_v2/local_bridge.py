from collections import deque
import math
import time

from .collision import collides_swept
from .dubins import plan_dubins_segment
from .geometry import wrap_pi
from .primitives import enumerate_actions, rollout_primitive


def _heading_bin(theta: float, bins: int):
    t = wrap_pi(theta) % (2.0 * math.pi)
    return int(t / (2.0 * math.pi) * bins) % bins


def _state_key(pose, gear: int, start_pose, step_m: float, heading_bins: int):
    sx, sy, _ = start_pose
    x, y, th = pose
    ix = int(round((x - sx) / step_m))
    iy = int(round((y - sy) / step_m))
    it = _heading_bin(th, heading_bins)
    return ix, iy, it, int(1 if gear >= 0 else -1)


def _merge_prefix_actions(actions, end_poses, cfg):
    cmds = []
    steps = []
    straight_acc = 0.0
    straight_mode = None
    straight_end = None
    turn_acc = 0.0
    turn_mode = None
    turn_end = None
    def flush_straight():
        nonlocal straight_acc, straight_mode, straight_end
        if straight_mode is None or straight_acc <= 0.0:
            straight_acc = 0.0
            straight_mode = None
            straight_end = None
            return
        cells = max(1, int(round(straight_acc / cfg.cell_size)))
        cmd = f"FW{cells * 10}" if straight_mode == "F" else f"BW{cells * 10}"
        cmds.append(cmd)
        steps.append((cmd, straight_end))
        straight_acc = 0.0
        straight_mode = None
        straight_end = None

    def flush_turn():
        nonlocal turn_acc, turn_mode, turn_end
        if turn_mode is None or turn_acc <= 0.0:
            turn_acc = 0.0
            turn_mode = None
            turn_end = None
            return
        deg = max(1, int(round(turn_acc)))
        cmd = f"{turn_mode}{deg:03d}"
        cmds.append(cmd)
        steps.append((cmd, turn_end))
        turn_acc = 0.0
        turn_mode = None
        turn_end = None

    for action, end_pose in zip(actions, end_poses):
        if action == "FS":
            flush_turn()
            if straight_mode == "R":
                flush_straight()
            straight_mode = "F"
            straight_acc += cfg.primitive_len
            straight_end = end_pose
            continue
        if action == "RS":
            flush_turn()
            if straight_mode == "F":
                flush_straight()
            straight_mode = "R"
            straight_acc += cfg.primitive_len
            straight_end = end_pose
            continue

        flush_straight()
        next_turn_mode = None
        next_turn_unit_deg = 0.0
        if action == "FL":
            next_turn_mode = "FL"
            next_turn_unit_deg = cfg.turn_unit_deg("L")
        elif action == "FR":
            next_turn_mode = "FR"
            next_turn_unit_deg = cfg.turn_unit_deg("R")
        elif action == "RL":
            next_turn_mode = "BL"
            next_turn_unit_deg = cfg.turn_unit_deg("L")
        elif action == "RR":
            next_turn_mode = "BR"
            next_turn_unit_deg = cfg.turn_unit_deg("R")

        if next_turn_mode is None:
            flush_turn()
            continue

        if turn_mode is not None and turn_mode != next_turn_mode:
            flush_turn()
        turn_mode = next_turn_mode
        turn_acc += next_turn_unit_deg
        turn_end = end_pose

    flush_straight()
    flush_turn()
    return cmds, steps


def _reconstruct_actions(node_idx: int, parents, actions):
    out = []
    cur = node_idx
    while cur > 0 and parents[cur] is not None:
        out.append(actions[cur])
        cur = parents[cur]
    out.reverse()
    return out


def _simulate_chain(start_pose, action_chain, cfg):
    cur = start_pose
    endpoints = []
    path = [start_pose]
    for act in action_chain:
        rollout = rollout_primitive(cur[0], cur[1], cur[2], act, cfg)
        cur = rollout.end_pose
        endpoints.append(cur)
        path.append(cur)
    return cur, endpoints, path


def _actions_from_commands(commands):
    actions = []
    for cmd in commands:
        if cmd.startswith("FW"):
            actions.append("FS")
        elif cmd.startswith("BW"):
            actions.append("RS")
        elif cmd.startswith("FL"):
            actions.append("FL")
        elif cmd.startswith("FR"):
            actions.append("FR")
        elif cmd.startswith("BL"):
            actions.append("RL")
        elif cmd.startswith("BR"):
            actions.append("RR")
    return actions


def _steer_reversal_count(actions):
    score = 0
    prev = None
    for act in actions:
        steer = act[1] if isinstance(act, str) and len(act) >= 2 else "S"
        if steer not in ("L", "R"):
            continue
        if prev in ("L", "R") and prev != steer:
            score += 1
        prev = steer
    return score


def _gear_switch_count(actions):
    score = 0
    prev = None
    for act in actions:
        if not isinstance(act, str) or len(act) < 1:
            continue
        gear = act[0]
        if gear not in ("F", "R"):
            continue
        if prev in ("F", "R") and prev != gear:
            score += 1
        prev = gear
    return score


def plan_local_bridge_segment(start_pose, goal_pose, obstacles_xy, cfg, hx, hy, deadline=None):
    if not bool(getattr(cfg, "local_bridge_enabled", True)):
        return {
            "success": False,
            "reason": "local_bridge_disabled",
            "path": [],
            "actions": [],
            "commands": [],
            "command_steps": [],
            "cost": float("inf"),
            "expanded": 0,
            "planner": "local_bridge",
        }

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
            "planner": "local_bridge",
        }

    step_m = max(0.02, float(getattr(cfg, "local_bridge_step_m", 0.10)))
    radius_m = max(step_m, float(getattr(cfg, "local_bridge_radius_m", 0.60)))
    heading_bins = max(8, int(getattr(cfg, "local_bridge_heading_bins", 16)))
    max_nodes = max(20, int(getattr(cfg, "local_bridge_max_nodes", 300)))
    dubins_every = max(1, int(getattr(cfg, "local_bridge_dubins_every", 3)))
    dubins_min_budget_s = max(0.0, float(getattr(cfg, "local_bridge_dubins_min_budget_ms", 120)) / 1000.0)
    allow_reverse = bool(getattr(cfg, "local_bridge_allow_reverse", True)) and bool(
        getattr(cfg, "reverse_enabled", True)
    )
    radius_sq = radius_m * radius_m

    poses = [start_pose]
    parents = [None]
    actions = [None]
    gears = [1]

    q = deque([0])
    visited = {_state_key(start_pose, 1, start_pose, step_m, heading_bins)}
    expanded = 0
    candidates = []
    best_score = None

    while q:
        if deadline is not None and time.monotonic() >= deadline:
            return {
                "success": False,
                "reason": "time_budget",
                "path": [],
                "actions": [],
                "commands": [],
                "command_steps": [],
                "cost": float("inf"),
                "expanded": expanded,
                "planner": "local_bridge",
            }

        idx = q.popleft()
        expanded += 1
        cur_pose = poses[idx]

        # Try analytic bridge from this local seed to the goal.
        should_try_dubins = expanded == 1 or (expanded % dubins_every == 0)
        if should_try_dubins:
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return {
                        "success": False,
                        "reason": "time_budget",
                        "path": [],
                        "actions": [],
                        "commands": [],
                        "command_steps": [],
                        "cost": float("inf"),
                        "expanded": expanded,
                        "planner": "local_bridge",
                    }
                if remaining < dubins_min_budget_s:
                    should_try_dubins = False
        if should_try_dubins:
            dubins_seg = plan_dubins_segment(cur_pose, goal_pose, obstacles_xy, cfg, hx, hy, deadline=deadline)
            if dubins_seg.get("success"):
                prefix_actions = _reconstruct_actions(idx, parents, actions)
                seed_pose, prefix_endpoints, prefix_path = _simulate_chain(start_pose, prefix_actions, cfg)
                if prefix_actions:
                    prefix_cmds, prefix_steps = _merge_prefix_actions(prefix_actions, prefix_endpoints, cfg)
                else:
                    prefix_cmds, prefix_steps = [], []

                bridge_steps = list(dubins_seg.get("command_steps", []))
                if bridge_steps:
                    bridge_steps[0] = (bridge_steps[0][0], bridge_steps[0][1])
                combined_steps = list(prefix_steps) + bridge_steps
                combined_cmds = [cmd for cmd, _ in combined_steps]
                combined_actions = list(prefix_actions) + _actions_from_commands(dubins_seg.get("commands", []))

                combined_path = list(prefix_path)
                dpath = dubins_seg.get("path", [])
                if dpath:
                    if combined_path and len(dpath) > 1:
                        combined_path.extend(dpath[1:])
                    elif not combined_path:
                        combined_path = list(dpath)
                if combined_path:
                    combined_path[-1] = goal_pose

                total_len = len(prefix_actions) * cfg.primitive_len + float(dubins_seg.get("cost", 0.0))
                steer_revs = _steer_reversal_count(combined_actions)
                gear_switch = _gear_switch_count(combined_actions)
                score = (total_len, steer_revs, gear_switch)
                candidates.append(
                    (
                        score,
                        {
                            "success": True,
                            "reason": None,
                            "path": combined_path,
                            "actions": combined_actions,
                            "commands": combined_cmds,
                            "command_steps": combined_steps,
                            "cost": total_len,
                            "expanded": expanded,
                            "planner": "local_bridge",
                        },
                    )
                )
                if best_score is None or score < best_score:
                    best_score = score
                if steer_revs == 0 and gear_switch <= 1 and len(prefix_actions) <= 2:
                    break
            elif dubins_seg.get("reason") == "time_budget":
                return {
                    "success": False,
                    "reason": "time_budget",
                    "path": [],
                    "actions": [],
                    "commands": [],
                    "command_steps": [],
                    "cost": float("inf"),
                    "expanded": expanded,
                    "planner": "local_bridge",
                }

        if len(poses) >= max_nodes:
            continue

        cur_gear = gears[idx]
        for act in enumerate_actions(cur_gear, allow_reverse):
            if deadline is not None and time.monotonic() >= deadline:
                return {
                    "success": False,
                    "reason": "time_budget",
                    "path": [],
                    "actions": [],
                    "commands": [],
                    "command_steps": [],
                    "cost": float("inf"),
                    "expanded": expanded,
                    "planner": "local_bridge",
                }
            rollout = rollout_primitive(cur_pose[0], cur_pose[1], cur_pose[2], act, cfg)
            if collides_swept(rollout.samples, obstacles_xy, cfg, hx, hy):
                continue
            nx, ny, nth = rollout.end_pose
            if (nx - start_pose[0]) * (nx - start_pose[0]) + (ny - start_pose[1]) * (ny - start_pose[1]) > radius_sq:
                continue

            key = _state_key(rollout.end_pose, rollout.next_gear, start_pose, step_m, heading_bins)
            if key in visited:
                continue

            visited.add(key)
            poses.append(rollout.end_pose)
            parents.append(idx)
            actions.append(act)
            gears.append(rollout.next_gear)
            q.append(len(poses) - 1)

    if not candidates:
        return {
            "success": False,
            "reason": "local_bridge_no_seed",
            "path": [],
            "actions": [],
            "commands": [],
            "command_steps": [],
            "cost": float("inf"),
            "expanded": expanded,
            "planner": "local_bridge",
        }

    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]
