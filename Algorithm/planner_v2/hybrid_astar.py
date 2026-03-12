import heapq
import math
import time

from .collision import collides_swept, min_obstacle_distance
from .geometry import theta_to_bin, wrap_pi
from .heuristic import heuristic
from .primitives import enumerate_actions, rollout_primitive


def _state_key(x: float, y: float, th: float, gear: int, last_steer: str, run_len_bin: int, cfg):
    ix = int(round(x / cfg.res_xy))
    iy = int(round(y / cfg.res_xy))
    it = theta_to_bin(th, cfg.n_theta)
    return (ix, iy, it, gear, last_steer, run_len_bin)


def _goal_reached(x: float, y: float, th: float, goal, cfg):
    gx, gy, gth = goal
    return math.hypot(gx - x, gy - y) <= cfg.pos_tol and abs(wrap_pi(gth - th)) <= cfg.theta_tol


def _try_simple_goal_connect(cur_pose, goal, obstacles_xy, cfg, hx, hy):
    # Lightweight analytic connector: short straight shot when heading aligns.
    x, y, th = cur_pose
    gx, gy, gth = goal
    dist_to_goal = math.hypot(gx - x, gy - y)
    if dist_to_goal > 2.0 * cfg.primitive_len:
        return None

    to_goal = math.atan2(gy - y, gx - x)
    if abs(wrap_pi(to_goal - th)) > math.radians(20):
        return None
    if abs(wrap_pi(gth - th)) > math.radians(25):
        return None

    # Keep connector length aligned with primitive steps so command serialization stays consistent.
    n_primitives = max(1, int(round(dist_to_goal / cfg.primitive_len)))
    shot_dist = n_primitives * cfg.primitive_len
    steps = max(1, int(round(shot_dist / cfg.substep_len)))
    ds = shot_dist / steps
    samples = []
    cx, cy = x, y
    for _ in range(steps):
        cx += ds * math.cos(th)
        cy += ds * math.sin(th)
        samples.append((cx, cy, th))

    if collides_swept(samples, obstacles_xy, cfg, hx, hy):
        return None

    end_pose = (cx, cy, th)
    if not _goal_reached(cx, cy, th, goal, cfg):
        return None

    return {
        "samples": samples,
        "end_pose": end_pose,
        "cost": shot_dist,
        "action": ("FS", n_primitives),
    }


def _steer_from_action(action):
    if isinstance(action, (tuple, list)):
        action = action[0]
    if not action or len(action) < 2:
        return "S"
    steer = action[1]
    if steer in ("L", "R"):
        return steer
    return "S"


def hybrid_astar_segment(start, goal, obstacles_xy, cfg, hx, hy, deadline=None, min_turn_run=0):
    sx, sy, sth = start
    gx, gy, gth = goal
    use_steer_memory = (min_turn_run > 0) or (getattr(cfg, "w_steer_switch", 0.0) > 0.0)

    sgear = 1
    start_k = _state_key(sx, sy, sth, sgear, "S", 0, cfg)

    gscore = {start_k: 0.0}
    pose_of = {start_k: (sx, sy, sth)}
    parent = {start_k: None}
    parent_action = {start_k: None}

    heap = []
    counter = 0
    h0 = heuristic(sx, sy, sth, gx, gy, gth, cfg)
    heapq.heappush(heap, (h0, counter, start_k, 0.0))

    best_goal_key = None
    expansions = 0

    while heap and expansions < cfg.max_expansions:
        if deadline is not None and time.monotonic() >= deadline:
            return {
                "success": False,
                "path": [],
                "actions": [],
                "cost": float("inf"),
                "expanded": expansions,
                "reason": "time_budget",
            }

        f, _, key, g_at_push = heapq.heappop(heap)
        if g_at_push != gscore.get(key):
            continue

        x, y, th = pose_of[key]
        cur_gear = key[3]
        cur_steer = key[4]
        cur_run_len = key[5]
        expansions += 1

        if _goal_reached(x, y, th, goal, cfg):
            best_goal_key = key
            break

        if cfg.analytic_expand_every > 0 and expansions % cfg.analytic_expand_every == 0:
            shot = _try_simple_goal_connect((x, y, th), goal, obstacles_xy, cfg, hx, hy)
            if shot is not None:
                ex, ey, eth = shot["end_pose"]
                skey = _state_key(ex, ey, eth, cur_gear, "S", 0, cfg)
                ng = gscore[key] + shot["cost"]
                if skey not in gscore or ng < gscore[skey]:
                    gscore[skey] = ng
                    pose_of[skey] = shot["end_pose"]
                    parent[skey] = key
                    parent_action[skey] = shot["action"]
                    best_goal_key = skey
                    break

        for action in enumerate_actions(
            cur_gear,
            cfg.reverse_enabled,
            bool(getattr(cfg, "allow_reverse_turn", True)),
        ):
            rollout = rollout_primitive(x, y, th, action, cfg)
            if collides_swept(rollout.samples, obstacles_xy, cfg, hx, hy):
                continue

            next_steer_raw = _steer_from_action(action)
            if use_steer_memory:
                if (
                    min_turn_run > 0
                    and cur_steer in ("L", "R")
                    and next_steer_raw in ("L", "R")
                    and cur_steer != next_steer_raw
                    and cur_run_len < min_turn_run
                ):
                    continue

                if next_steer_raw == "S":
                    next_run_len = 0
                elif next_steer_raw == cur_steer:
                    next_run_len = min(cur_run_len + 1, 2)
                else:
                    next_run_len = 1
                next_steer = next_steer_raw
            else:
                next_steer = "S"
                next_run_len = 0

            nx, ny, nth = rollout.end_pose
            nk = _state_key(nx, ny, nth, rollout.next_gear, next_steer, next_run_len, cfg)

            turn_cost = cfg.w_turn if action[1] in ("L", "R") else 0.0
            reverse_cost = cfg.w_reverse if rollout.next_gear < 0 else 0.0
            switch_cost = cfg.w_switch if rollout.next_gear != cur_gear else 0.0
            steer_switch_cost = 0.0
            if use_steer_memory:
                steer_switch_cost = (
                    cfg.w_steer_switch
                    if cur_steer in ("L", "R") and next_steer in ("L", "R") and cur_steer != next_steer
                    else 0.0
                )
            clearance = min_obstacle_distance(nx, ny, obstacles_xy, cfg.obs_size / 2.0)
            clearance_cost = cfg.w_clearance * (1.0 / (1e-3 + clearance))

            step_cost = (
                rollout.cost_len
                + turn_cost
                + reverse_cost
                + switch_cost
                + steer_switch_cost
                + clearance_cost
            )
            ng = gscore[key] + step_cost

            if nk not in gscore or ng < gscore[nk]:
                gscore[nk] = ng
                pose_of[nk] = (nx, ny, nth)
                parent[nk] = key
                parent_action[nk] = action
                counter += 1
                h = heuristic(nx, ny, nth, gx, gy, gth, cfg)
                heapq.heappush(heap, (ng + h, counter, nk, ng))

    if best_goal_key is None:
        return {
            "success": False,
            "path": [],
            "actions": [],
            "cost": float("inf"),
            "expanded": expansions,
            "reason": "no_path",
        }

    rev_path = []
    rev_actions = []
    cur = best_goal_key
    while cur is not None:
        rev_path.append(pose_of[cur])
        act = parent_action[cur]
        if act is not None:
            rev_actions.append(act)
        cur = parent[cur]

    rev_path.reverse()
    rev_actions.reverse()

    return {
        "success": True,
        "path": rev_path,
        "actions": rev_actions,
        "cost": gscore[best_goal_key],
        "expanded": expansions,
        "reason": None,
    }
