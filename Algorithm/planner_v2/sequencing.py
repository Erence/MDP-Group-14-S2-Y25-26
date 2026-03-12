from dataclasses import replace
import math
import re
import time

from .dubins import plan_dubins_segment
from .dubins_dock import plan_dubins_dock_segment
from .hybrid_astar import hybrid_astar_segment
from .local_bridge import plan_local_bridge_segment
from .reeds_shepp import plan_reeds_shepp_segment
from .view_states import generate_view_states


def _planner_mode_flags(mode: str):
    normalized = str(mode or "dubins_fallback").strip().lower()
    if normalized == "dubins_only":
        return True, False
    if normalized == "hybrid_only":
        return False, True
    return True, True


def _steer_reversal_score_actions(actions):
    score = 0
    prev = None
    for action in actions:
        name = action[0] if isinstance(action, (tuple, list)) else action
        if not isinstance(name, str) or len(name) < 2:
            continue
        steer = name[1]
        if steer not in ("L", "R"):
            continue
        if prev in ("L", "R") and steer != prev:
            score += 1
        prev = steer
    return score


_TURN_CMD_PATTERN = re.compile(r"^(FR|FL|BR|BL)(\d+)$")
_FORWARD_EPS = 1e-6


def _turn_radians_from_motion_command(cmd):
    m = _TURN_CMD_PATTERN.fullmatch(str(cmd).strip())
    if not m:
        return None
    angle_deg = int(m.group(2))
    if angle_deg <= 0:
        return 0.0
    return math.radians(angle_deg)


def _segment_turn_radians(seg, cfg):
    def _sum_turn_from_cmd_items(items):
        total = 0.0
        found = False
        for item in items or []:
            cmd = item[0] if isinstance(item, (tuple, list)) and item else item
            turn_rad = _turn_radians_from_motion_command(cmd)
            if turn_rad is None:
                continue
            total += turn_rad
            found = True
        return found, total

    found, total = _sum_turn_from_cmd_items(seg.get("command_steps", []))
    if found:
        return total

    found, total = _sum_turn_from_cmd_items(seg.get("commands", []))
    if found:
        return total

    total = 0.0
    turn_left_rad = math.radians(cfg.turn_unit_deg("L"))
    turn_right_rad = math.radians(cfg.turn_unit_deg("R"))
    for action in seg.get("actions", []):
        name = action[0] if isinstance(action, (tuple, list)) else action
        if name in ("FL", "RL"):
            total += turn_left_rad
        elif name in ("FR", "RR"):
            total += turn_right_rad
    return total


def _leg_effective_components(seg, cfg):
    base_cost = float(seg.get("cost", float("inf")))
    w_leg_distance_quad = max(0.0, float(getattr(cfg, "w_leg_distance_quad", 0.0)))
    w_leg_turn_quad = max(0.0, float(getattr(cfg, "w_leg_turn_quad", 0.0)))

    if not math.isfinite(base_cost):
        return {
            "base_cost": base_cost,
            "distance_quad_penalty": float("inf"),
            "turn_rad": 0.0,
            "turn_quad_penalty": 0.0,
            "effective_cost": float("inf"),
        }

    turn_rad = _segment_turn_radians(seg, cfg)
    distance_quad_penalty = w_leg_distance_quad * (base_cost ** 2)
    turn_quad_penalty = w_leg_turn_quad * (turn_rad ** 2)
    effective_cost = base_cost + distance_quad_penalty + turn_quad_penalty
    return {
        "base_cost": base_cost,
        "distance_quad_penalty": distance_quad_penalty,
        "turn_rad": turn_rad,
        "turn_quad_penalty": turn_quad_penalty,
        "effective_cost": effective_cost,
    }


def _base_debug(cfg, candidate_count: int):
    w_leg_distance_quad = max(0.0, float(getattr(cfg, "w_leg_distance_quad", 0.0)))
    w_leg_turn_quad = max(0.0, float(getattr(cfg, "w_leg_turn_quad", 0.0)))
    return {
        "planner_mode": getattr(cfg, "planner_mode", "dubins_fallback"),
        "sequence_mode": getattr(cfg, "sequence_mode", "greedy_nearest"),
        "start_straight_bias": bool(getattr(cfg, "start_straight_bias", False)),
        "leg_penalty_weights": {
            "w_leg_distance_quad": w_leg_distance_quad,
            "w_leg_turn_quad": w_leg_turn_quad,
        },
        "fallback_used_count": 0,
        "dubins_used_count": 0,
        "local_bridge_used_count": 0,
        "reeds_shepp_used_count": 0,
        "attempted_expansions_total": 0,
        "expanded_total": 0,
        "candidate_count": candidate_count,
        "leg_planners": [],
        "connector_used_per_leg": [],
        "connector_attempts_per_leg": [],
        "connector_reason_per_leg": [],
        "leg_time_ms": [],
        "rs_word_per_leg": [],
        "smoothing_retries_used": 0,
        "time_budget_hit": False,
        "micro_tweak_score_per_leg": [],
        "best_effort_returned": False,
        "partial": False,
        "partial_reason": None,
        "completed_obstacles": 0,
        "total_obstacles": candidate_count,
        "remaining_obstacle_ids": [],
        "skipped_no_view_state_ids": [],
        "skipped_unreachable_ids": [],
        "skipped_obstacle_ids": [],
        "leg_effective_components": [],
        "effective_cost_total": 0.0,
    }


def _finalize_seg(seg, attempts, start_ts):
    out = dict(seg)
    out["connector_attempts"] = attempts
    fail_reasons = [a.get("reason") for a in attempts if not a.get("success") and a.get("reason")]
    out["connector_reason"] = fail_reasons[-1] if fail_reasons else out.get("reason")
    out["planning_ms"] = round((time.monotonic() - start_ts) * 1000.0, 3)
    if out.get("planner") == "reeds_shepp":
        out["rs_word"] = out.get("word", "")
    return out


def _plan_leg(from_pose, to_pose, obstacles_xy, cfg, hx, hy, deadline=None):
    connector_order = str(getattr(cfg, "connector_order", "dubins_local_rs_hybrid")).strip().lower()
    if connector_order != "dubins_local_rs_hybrid":
        seg = {
            "success": False,
            "reason": "unsupported_connector_order",
            "path": [],
            "actions": [],
            "cost": float("inf"),
            "expanded": 0,
            "planner": "fallback",
        }
        return _finalize_seg(seg, [], time.monotonic()), False, 0, 0, False

    start_ts = time.monotonic()
    leg_slice_s = max(0.0, float(getattr(cfg, "leg_time_slice_s", 0.0)))
    active_deadline = deadline
    if leg_slice_s > 0.0:
        leg_deadline = start_ts + leg_slice_s
        active_deadline = leg_deadline if deadline is None else min(deadline, leg_deadline)
    rs_min_budget_s = max(0.0, float(getattr(cfg, "rs_min_budget_ms", 250)) / 1000.0)

    def _remaining_budget_s():
        if active_deadline is None:
            return None
        return max(0.0, active_deadline - time.monotonic())

    use_dubins, use_fallback = _planner_mode_flags(getattr(cfg, "planner_mode", "dubins_fallback"))
    smooth_mode = str(getattr(cfg, "smooth_mode", "max") or "max").strip().lower()
    attempts = []

    def record_attempt(planner_name: str, seg):
        attempts.append(
            {
                "planner": planner_name,
                "success": bool(seg.get("success")),
                "reason": seg.get("reason"),
            }
        )

    if active_deadline is not None and time.monotonic() >= active_deadline:
        seg = {
            "success": False,
            "reason": "time_budget",
            "path": [],
            "actions": [],
            "cost": float("inf"),
            "expanded": 0,
            "planner": "fallback",
        }
        return _finalize_seg(seg, attempts, start_ts), False, 0, 0, True

    dubins_seg = None
    if use_dubins:
        dubins_seg = plan_dubins_segment(from_pose, to_pose, obstacles_xy, cfg, hx, hy, deadline=active_deadline)
        record_attempt("dubins", dubins_seg)
        if dubins_seg.get("success"):
            return _finalize_seg(dubins_seg, attempts, start_ts), False, 0, 0, False
        if dubins_seg.get("reason") == "time_budget":
            return _finalize_seg(dubins_seg, attempts, start_ts), False, 0, 0, True

        dock_seg = plan_dubins_dock_segment(from_pose, to_pose, obstacles_xy, cfg, hx, hy, deadline=active_deadline)
        record_attempt("dubins_dock", dock_seg)
        if dock_seg.get("success"):
            return _finalize_seg(dock_seg, attempts, start_ts), False, 0, 0, False
        if dock_seg.get("reason") == "time_budget":
            return _finalize_seg(dock_seg, attempts, start_ts), False, 0, 0, True

        local_seg = plan_local_bridge_segment(from_pose, to_pose, obstacles_xy, cfg, hx, hy, deadline=active_deadline)
        record_attempt("local_bridge", local_seg)
        if local_seg.get("success"):
            return _finalize_seg(local_seg, attempts, start_ts), False, 0, 0, False
        if local_seg.get("reason") == "time_budget":
            return _finalize_seg(local_seg, attempts, start_ts), False, 0, 0, True

        if bool(getattr(cfg, "rs_enabled", True)) and bool(getattr(cfg, "reverse_enabled", True)):
            remaining_s = _remaining_budget_s()
            if remaining_s is not None and remaining_s < rs_min_budget_s:
                rs_seg = {
                    "success": False,
                    "reason": "rs_skipped_low_budget",
                    "path": [],
                    "actions": [],
                    "cost": float("inf"),
                    "expanded": 0,
                    "planner": "reeds_shepp",
                }
            else:
                rs_seg = plan_reeds_shepp_segment(
                    from_pose,
                    to_pose,
                    obstacles_xy,
                    cfg,
                    hx,
                    hy,
                    deadline=active_deadline,
                )
            record_attempt("reeds_shepp", rs_seg)
            if rs_seg.get("success"):
                return _finalize_seg(rs_seg, attempts, start_ts), False, 0, 0, False
            if rs_seg.get("reason") == "time_budget":
                return _finalize_seg(rs_seg, attempts, start_ts), False, 0, 0, True

    if not use_fallback:
        reason = "no_dubins_path"
        if dubins_seg is not None:
            reason = dubins_seg.get("reason", reason)
        seg = {
            "success": False,
            "reason": reason,
            "path": [],
            "actions": [],
            "cost": float("inf"),
            "expanded": 0,
            "planner": "fallback",
        }
        return _finalize_seg(seg, attempts, start_ts), False, 0, 0, False

    fallback_cfg = replace(cfg, reverse_enabled=bool(getattr(cfg, "reverse_enabled", True)))
    retry_levels = max(1, int(getattr(cfg, "hybrid_retry_levels", 1)))
    candidates = []
    attempted_expanded = 0
    retries_used = 0
    time_budget_hit = False

    for level in range(retry_levels):
        if active_deadline is not None and time.monotonic() >= active_deadline:
            time_budget_hit = True
            break

        min_turn_run = (
            int(getattr(cfg, "min_turn_run_strict", 2))
            if level == 0
            else int(getattr(cfg, "min_turn_run_relaxed", 1))
        )
        seg = hybrid_astar_segment(
            from_pose,
            to_pose,
            obstacles_xy,
            fallback_cfg,
            hx,
            hy,
            deadline=active_deadline,
            min_turn_run=min_turn_run,
        )
        seg["planner"] = "hybrid_strict" if level == 0 else "hybrid_relaxed"
        record_attempt(seg["planner"], seg)
        attempted_expanded += seg.get("expanded", 0)

        if seg.get("reason") == "time_budget":
            time_budget_hit = True
            break

        if seg.get("success"):
            if seg.get("path"):
                # Snap fallback endpoint to the selected target pose for stable multi-leg chaining.
                seg["path"][-1] = to_pose
            seg["_smooth_score"] = _steer_reversal_score_actions(seg.get("actions", []))
            seg["_retry_level"] = level
            candidates.append(seg)
            if smooth_mode != "max" or (level == 0 and seg["_smooth_score"] == 0):
                break
            if level > 0:
                break
        retries_used = max(retries_used, level)

    if not candidates:
        reason = "time_budget" if time_budget_hit else "no_path"
        seg = {
            "success": False,
            "reason": reason,
            "path": [],
            "actions": [],
            "cost": float("inf"),
            "expanded": attempted_expanded,
            "planner": "fallback",
        }
        return _finalize_seg(seg, attempts, start_ts), True, attempted_expanded, retries_used, time_budget_hit

    candidates.sort(
        key=lambda seg: (
            seg.get("_smooth_score", 0),
            seg.get("cost", float("inf")),
            len(seg.get("actions", [])),
        )
    )
    best = candidates[0]
    best.pop("_smooth_score", None)
    best.pop("_retry_level", None)
    return _finalize_seg(best, attempts, start_ts), True, attempted_expanded, retries_used, time_budget_hit


def plan_sequence(start_pose, obstacles, cfg, hx, hy, deadline=None):
    obs_all = [{**ob} for ob in obstacles]
    all_obs_xy = [(ob["x_m"], ob["y_m"]) for ob in obs_all]
    for ob in obs_all:
        ob["all_obstacles_m"] = obs_all

    targets = []
    skipped_no_view_state_ids = []
    for oi, ob in enumerate(obs_all):
        obstacle_id = ob.get("id", oi)
        cands = generate_view_states(ob, cfg, hx, hy)
        if not cands:
            skipped_no_view_state_ids.append(obstacle_id)
            continue
        targets.append(
            {
                "obstacle_index": oi,
                "obstacle_id": obstacle_id,
                "pose": cands[0],
            }
        )

    n_obs = len(targets)
    total_obstacles = len(obs_all)
    if n_obs == 0:
        partial = bool(skipped_no_view_state_ids)
        partial_reason = "skipped_no_view_state" if partial else None
        skipped_obstacle_ids = list(skipped_no_view_state_ids)
        return {
            "success": True,
            "visit_order": [],
            "selected": [],
            "segments": [],
            "cost": 0.0,
            "debug": {
                **_base_debug(cfg, n_obs),
                "best_effort_returned": partial,
                "partial": partial,
                "partial_reason": partial_reason,
                "completed_obstacles": 0,
                "total_obstacles": total_obstacles,
                "remaining_obstacle_ids": [],
                "skipped_no_view_state_ids": skipped_no_view_state_ids,
                "skipped_unreachable_ids": [],
                "skipped_obstacle_ids": skipped_obstacle_ids,
            },
        }

    if str(getattr(cfg, "sequence_mode", "greedy_nearest")).strip().lower() != "greedy_nearest":
        return {
            "success": False,
            "reason": "unsupported_sequence_mode",
            "debug": _base_debug(cfg, n_obs),
        }

    attempted_expansions_total = 0
    smoothing_retries_used = 0
    time_budget_hit = False
    leg_cache = {}

    def _pose_of(index: int):
        if index == -1:
            return start_pose
        return targets[index]["pose"]

    def _cached_leg(from_idx: int, to_idx: int):
        nonlocal attempted_expansions_total, smoothing_retries_used, time_budget_hit
        key = (from_idx, to_idx)
        if key in leg_cache:
            return leg_cache[key]
        seg, fallback_attempted, fallback_expanded, retries_used, budget_hit = _plan_leg(
            _pose_of(from_idx), _pose_of(to_idx), all_obs_xy, cfg, hx, hy, deadline=deadline
        )
        if fallback_attempted:
            attempted_expansions_total += fallback_expanded
            smoothing_retries_used += retries_used
        if budget_hit:
            time_budget_hit = True
        leg_cache[key] = seg
        return seg

    def _ordered_candidates(cur_idx: int, remaining):
        from_pose = _pose_of(cur_idx)
        use_start_bias = bool(getattr(cfg, "start_straight_bias", False)) and cur_idx == -1
        fx, fy, fth = from_pose
        heading_x = math.cos(fth)
        heading_y = math.sin(fth)
        ranked = []
        for idx in remaining:
            to_pose = _pose_of(idx)
            dx = to_pose[0] - fx
            dy = to_pose[1] - fy
            dist_sq = dx * dx + dy * dy
            if use_start_bias:
                # Rank by: in-front first, low lateral error, small heading change, then distance.
                forward = dx * heading_x + dy * heading_y
                lateral = abs(dx * heading_y - dy * heading_x)
                heading_delta = abs((to_pose[2] - fth + math.pi) % (2.0 * math.pi) - math.pi)
                tier = 0 if forward > _FORWARD_EPS else 1
                key = (tier, lateral, heading_delta, dist_sq)
            else:
                tier = 0
                key = (dist_sq,)
            ranked.append({"idx": idx, "tier": tier, "key": key})
        ranked.sort(key=lambda item: item["key"])
        return ranked

    def _greedy_route(use_cache: bool, active_cfg, active_deadline):
        nonlocal attempted_expansions_total, smoothing_retries_used, time_budget_hit
        cur_idx = -1
        remaining = set(range(n_obs))
        order_indices = []
        segments = []
        total_cost = 0.0

        while remaining:
            if active_deadline is not None and time.monotonic() >= active_deadline:
                time_budget_hit = True
                break

            candidates = []
            ranked_candidates = _ordered_candidates(cur_idx, remaining)
            for ranked in ranked_candidates:
                idx = ranked["idx"]
                if use_cache:
                    seg = _cached_leg(cur_idx, idx)
                else:
                    seg, fallback_attempted, fallback_expanded, retries_used, budget_hit = _plan_leg(
                        _pose_of(cur_idx),
                        _pose_of(idx),
                        all_obs_xy,
                        active_cfg,
                        hx,
                        hy,
                        deadline=active_deadline,
                    )
                    if fallback_attempted:
                        attempted_expansions_total += fallback_expanded
                        smoothing_retries_used += retries_used
                    if budget_hit:
                        time_budget_hit = True

                if seg.get("reason") == "time_budget":
                    time_budget_hit = True
                if seg.get("success"):
                    components = _leg_effective_components(seg, active_cfg)
                    candidates.append(
                        {
                            "idx": idx,
                            "tier": ranked["tier"],
                            "rank_key": ranked["key"],
                            "seg": seg,
                            "components": components,
                        }
                    )

            if not candidates:
                break

            if bool(getattr(active_cfg, "start_straight_bias", False)) and cur_idx == -1:
                min_tier = min(c["tier"] for c in candidates)
                candidates = [c for c in candidates if c["tier"] == min_tier]

            picked = min(
                candidates,
                key=lambda c: (
                    c["components"]["effective_cost"],
                    c["components"]["base_cost"],
                    c["rank_key"],
                    c["idx"],
                ),
            )

            idx = picked["idx"]
            seg = dict(picked["seg"])
            seg["_effective_components"] = picked["components"]
            order_indices.append(idx)
            segments.append(seg)
            total_cost += seg.get("cost", 0.0)
            remaining.remove(idx)
            cur_idx = idx

        return (len(remaining) == 0), order_indices, segments, total_cost, remaining

    reached_all, order_indices, segments, total_cost, remaining_indices = _greedy_route(True, cfg, deadline)
    skipped_unreachable_ids = [targets[idx]["obstacle_id"] for idx in sorted(remaining_indices)]

    skipped_obstacle_ids = list(skipped_no_view_state_ids)
    for obstacle_id in skipped_unreachable_ids:
        if obstacle_id not in skipped_obstacle_ids:
            skipped_obstacle_ids.append(obstacle_id)

    partial = bool(skipped_obstacle_ids) or time_budget_hit
    partial_reason = None
    if time_budget_hit:
        partial_reason = "time_budget"
    elif skipped_unreachable_ids:
        partial_reason = "skipped_unreachable"
    elif skipped_no_view_state_ids:
        partial_reason = "skipped_no_view_state"

    best_effort_returned = partial and (not reached_all or bool(skipped_no_view_state_ids))

    selected = []
    visit_order = []
    fallback_used_count = 0
    dubins_used_count = 0
    local_bridge_used_count = 0
    reeds_shepp_used_count = 0
    expanded_total = 0
    leg_planners = []
    connector_used_per_leg = []
    connector_attempts_per_leg = []
    connector_reason_per_leg = []
    leg_time_ms = []
    rs_word_per_leg = []
    micro_tweak_score_per_leg = []
    leg_effective_components = []
    effective_cost_total = 0.0

    for idx, seg in zip(order_indices, segments):
        chosen = targets[idx]
        selected.append(
            {
                "obstacle_index": chosen["obstacle_index"],
                "obstacle_id": chosen["obstacle_id"],
                "pose": chosen["pose"],
            }
        )
        visit_order.append(chosen["obstacle_id"])
        expanded_total += seg.get("expanded", 0)

        planner_name = seg.get("planner", "fallback")
        connector_used_per_leg.append(planner_name)
        connector_attempts_per_leg.append(seg.get("connector_attempts", []))
        connector_reason_per_leg.append(seg.get("connector_reason"))
        leg_time_ms.append(seg.get("planning_ms", 0.0))
        rs_word_per_leg.append(seg.get("rs_word") if planner_name == "reeds_shepp" else None)
        micro_tweak_score_per_leg.append(_steer_reversal_score_actions(seg.get("actions", [])))
        components = seg.get("_effective_components")
        if components is None:
            components = _leg_effective_components(seg, cfg)
        leg_effective_components.append(
            {
                "base_cost": float(components["base_cost"]),
                "distance_quad_penalty": float(components["distance_quad_penalty"]),
                "turn_rad": float(components["turn_rad"]),
                "turn_quad_penalty": float(components["turn_quad_penalty"]),
                "effective_cost": float(components["effective_cost"]),
            }
        )
        effective_cost_total += float(components["effective_cost"])

        if planner_name in ("dubins", "dubins_dock"):
            dubins_used_count += 1
            leg_planners.append(planner_name)
        elif planner_name == "local_bridge":
            local_bridge_used_count += 1
            leg_planners.append("local_bridge")
        elif planner_name == "reeds_shepp":
            reeds_shepp_used_count += 1
            leg_planners.append("reeds_shepp")
        else:
            fallback_used_count += 1
            leg_planners.append(planner_name)

    visited_indices = set(order_indices)
    return {
        "success": True,
        "visit_order": visit_order,
        "selected": selected,
        "segments": segments,
        "cost": total_cost,
        "debug": {
            "planner_mode": getattr(cfg, "planner_mode", "dubins_fallback"),
            "sequence_mode": getattr(cfg, "sequence_mode", "greedy_nearest"),
            "start_straight_bias": bool(getattr(cfg, "start_straight_bias", False)),
            "leg_penalty_weights": {
                "w_leg_distance_quad": max(0.0, float(getattr(cfg, "w_leg_distance_quad", 0.0))),
                "w_leg_turn_quad": max(0.0, float(getattr(cfg, "w_leg_turn_quad", 0.0))),
            },
            "fallback_used_count": fallback_used_count,
            "dubins_used_count": dubins_used_count,
            "local_bridge_used_count": local_bridge_used_count,
            "reeds_shepp_used_count": reeds_shepp_used_count,
            "attempted_expansions_total": attempted_expansions_total,
            "expanded_total": expanded_total,
            "candidate_count": n_obs,
            "leg_planners": leg_planners,
            "connector_used_per_leg": connector_used_per_leg,
            "connector_attempts_per_leg": connector_attempts_per_leg,
            "connector_reason_per_leg": connector_reason_per_leg,
            "leg_time_ms": leg_time_ms,
            "rs_word_per_leg": rs_word_per_leg,
            "smoothing_retries_used": smoothing_retries_used,
            "time_budget_hit": time_budget_hit,
            "micro_tweak_score_per_leg": micro_tweak_score_per_leg,
            "best_effort_returned": best_effort_returned,
            "partial": partial,
            "partial_reason": partial_reason,
            "completed_obstacles": len(visit_order),
            "total_obstacles": total_obstacles,
            "remaining_obstacle_ids": [
                targets[idx]["obstacle_id"] for idx in range(n_obs) if idx not in visited_indices
            ],
            "skipped_no_view_state_ids": skipped_no_view_state_ids,
            "skipped_unreachable_ids": skipped_unreachable_ids,
            "skipped_obstacle_ids": skipped_obstacle_ids,
            "leg_effective_components": leg_effective_components,
            "effective_cost_total": effective_cost_total,
        },
    }
