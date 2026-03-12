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
_VERTICAL_BIAS_LOW_MAX_RATIO = 1.0 / 3.0
_VERTICAL_BIAS_MID_MAX_RATIO = 2.0 / 3.0
_HORIZONTAL_BIAS_LOW_MAX_RATIO = 1.0 / 3.0
_HORIZONTAL_BIAS_MID_MAX_RATIO = 2.0 / 3.0


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
    turn_left_forward_rad = math.radians(cfg.turn_unit_deg("L", gear=1))
    turn_right_forward_rad = math.radians(cfg.turn_unit_deg("R", gear=1))
    turn_left_reverse_rad = math.radians(cfg.turn_unit_deg("L", gear=-1))
    turn_right_reverse_rad = math.radians(cfg.turn_unit_deg("R", gear=-1))
    for action in seg.get("actions", []):
        name = action[0] if isinstance(action, (tuple, list)) else action
        if name == "FL":
            total += turn_left_forward_rad
        elif name == "FR":
            total += turn_right_forward_rad
        elif name == "RL":
            total += turn_left_reverse_rad
        elif name == "RR":
            total += turn_right_reverse_rad
    return total


def _segment_has_backward_turn(seg):
    def _iter_cmd_like(items):
        for item in items or []:
            cmd = item[0] if isinstance(item, (tuple, list)) and item else item
            if isinstance(cmd, str):
                yield cmd.strip().upper()

    for cmd in _iter_cmd_like(seg.get("command_steps", [])):
        if cmd.startswith("BL") or cmd.startswith("BR"):
            return True
    for cmd in _iter_cmd_like(seg.get("commands", [])):
        if cmd.startswith("BL") or cmd.startswith("BR"):
            return True
    for action in seg.get("actions", []):
        name = action[0] if isinstance(action, (tuple, list)) else action
        if name in ("RL", "RR"):
            return True
    return False


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
    vertical_bias_bands_m = {
        "low": float(getattr(cfg, "capture_vertical_bias_low_m", 0.0)),
        "mid": float(getattr(cfg, "capture_vertical_bias_mid_m", 0.0)),
        "high": float(getattr(cfg, "capture_vertical_bias_high_m", 0.0)),
    }
    horizontal_bias_bands_m = {
        "low": float(getattr(cfg, "capture_horizontal_bias_low_m", 0.0)),
        "mid": float(getattr(cfg, "capture_horizontal_bias_mid_m", 0.0)),
        "high": float(getattr(cfg, "capture_horizontal_bias_high_m", 0.0)),
    }
    return {
        "planner_mode": getattr(cfg, "planner_mode", "dubins_fallback"),
        "sequence_mode": getattr(cfg, "sequence_mode", "greedy_nearest"),
        "start_straight_bias": bool(getattr(cfg, "start_straight_bias", False)),
        "allow_reverse_turn": bool(getattr(cfg, "allow_reverse_turn", True)),
        "horizontal_bias_mode": "current_x_distance",
        "horizontal_bias_reference": "per_leg_current_x",
        "vertical_bias_bands_m": vertical_bias_bands_m,
        "vertical_bias_band_splits_ratio": {
            "low_max": _VERTICAL_BIAS_LOW_MAX_RATIO,
            "mid_max": _VERTICAL_BIAS_MID_MAX_RATIO,
        },
        "horizontal_bias_bands_m": horizontal_bias_bands_m,
        "horizontal_bias_band_splits_ratio": {
            "low_max": _HORIZONTAL_BIAS_LOW_MAX_RATIO,
            "mid_max": _HORIZONTAL_BIAS_MID_MAX_RATIO,
        },
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
        "obstacle_vertical_bias": [],
        "obstacle_horizontal_bias": [],
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
    allow_reverse_turn = bool(getattr(cfg, "allow_reverse_turn", True))

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
        if dubins_seg.get("success") and (not allow_reverse_turn) and _segment_has_backward_turn(dubins_seg):
            dubins_seg = {**dubins_seg, "success": False, "reason": "reverse_turn_disabled"}
        record_attempt("dubins", dubins_seg)
        if dubins_seg.get("success"):
            return _finalize_seg(dubins_seg, attempts, start_ts), False, 0, 0, False
        if dubins_seg.get("reason") == "time_budget":
            return _finalize_seg(dubins_seg, attempts, start_ts), False, 0, 0, True

        dock_seg = plan_dubins_dock_segment(from_pose, to_pose, obstacles_xy, cfg, hx, hy, deadline=active_deadline)
        if dock_seg.get("success") and (not allow_reverse_turn) and _segment_has_backward_turn(dock_seg):
            dock_seg = {**dock_seg, "success": False, "reason": "reverse_turn_disabled"}
        record_attempt("dubins_dock", dock_seg)
        if dock_seg.get("success"):
            return _finalize_seg(dock_seg, attempts, start_ts), False, 0, 0, False
        if dock_seg.get("reason") == "time_budget":
            return _finalize_seg(dock_seg, attempts, start_ts), False, 0, 0, True

        local_seg = plan_local_bridge_segment(from_pose, to_pose, obstacles_xy, cfg, hx, hy, deadline=active_deadline)
        if local_seg.get("success") and (not allow_reverse_turn) and _segment_has_backward_turn(local_seg):
            local_seg = {**local_seg, "success": False, "reason": "reverse_turn_disabled"}
        record_attempt("local_bridge", local_seg)
        if local_seg.get("success"):
            return _finalize_seg(local_seg, attempts, start_ts), False, 0, 0, False
        if local_seg.get("reason") == "time_budget":
            return _finalize_seg(local_seg, attempts, start_ts), False, 0, 0, True

        if (
            bool(getattr(cfg, "rs_enabled", True))
            and bool(getattr(cfg, "reverse_enabled", True))
            and allow_reverse_turn
        ):
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
        elif bool(getattr(cfg, "rs_enabled", True)) and bool(getattr(cfg, "reverse_enabled", True)):
            rs_skipped = {
                "success": False,
                "reason": "reverse_turn_disabled",
                "path": [],
                "actions": [],
                "cost": float("inf"),
                "expanded": 0,
                "planner": "reeds_shepp",
            }
            record_attempt("reeds_shepp", rs_skipped)

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
            if not bool(getattr(cfg, "allow_reverse_turn", True)) and _segment_has_backward_turn(seg):
                continue
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

    total_obstacles = len(obs_all)

    if str(getattr(cfg, "sequence_mode", "greedy_nearest")).strip().lower() != "greedy_nearest":
        return {
            "success": False,
            "reason": "unsupported_sequence_mode",
            "debug": _base_debug(cfg, total_obstacles),
        }

    attempted_expansions_total = 0
    smoothing_retries_used = 0
    time_budget_hit = False
    remaining = set(range(total_obstacles))
    order_indices = []
    selected = []
    segments = []
    total_cost = 0.0
    cur_pose = start_pose
    obstacle_vertical_bias_by_index = {}
    obstacle_horizontal_bias_by_index = {}

    def _record_bias_debug(index: int, has_view_state: bool):
        ob = obs_all[index]
        obstacle_id = ob.get("id", index)
        obstacle_vertical_bias_by_index[index] = {
            "obstacle_id": obstacle_id,
            "face_dir": ob.get("face_dir"),
            "vertical_bias_band": ob.get("_vertical_bias_band"),
            "vertical_bias_applied_m": float(ob.get("_vertical_bias_applied_m", 0.0)),
            "vertical_bias_y_ratio": ob.get("_vertical_bias_y_ratio"),
            "has_view_state": bool(has_view_state),
        }
        obstacle_horizontal_bias_by_index[index] = {
            "obstacle_id": obstacle_id,
            "face_dir": ob.get("face_dir"),
            "reference_x_m": float(ob.get("_horizontal_bias_reference_x_m", ob["x_m"])),
            "horizontal_distance_x_m": float(ob.get("_horizontal_bias_distance_x_m", 0.0)),
            "horizontal_bias_band": ob.get("_horizontal_bias_band"),
            "horizontal_bias_applied_m": float(ob.get("_horizontal_bias_applied_m", 0.0)),
            "horizontal_bias_ratio": ob.get("_horizontal_bias_x_ratio"),
            "has_view_state": bool(has_view_state),
        }

    def _compute_targets(reference_pose):
        reference_x = reference_pose[0]
        targets = {}
        for idx in remaining:
            ob = obs_all[idx]
            obstacle_id = ob.get("id", idx)
            cands = generate_view_states(ob, cfg, hx, hy, reference_x_m=reference_x)
            _record_bias_debug(idx, bool(cands))
            if not cands:
                continue
            targets[idx] = {
                "obstacle_index": idx,
                "obstacle_id": obstacle_id,
                "pose": cands[0],
            }
        return targets

    def _ordered_candidates(from_pose, targets, is_first_leg: bool):
        use_start_bias = bool(getattr(cfg, "start_straight_bias", False)) and is_first_leg
        fx, fy, fth = from_pose
        heading_x = math.cos(fth)
        heading_y = math.sin(fth)
        ranked = []
        for idx, target in targets.items():
            to_pose = target["pose"]
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
            ranked.append({"idx": idx, "tier": tier, "key": key, "target": target})
        ranked.sort(key=lambda item: item["key"])
        return ranked

    while remaining:
        if deadline is not None and time.monotonic() >= deadline:
            time_budget_hit = True
            break

        targets = _compute_targets(cur_pose)
        ranked_candidates = _ordered_candidates(cur_pose, targets, is_first_leg=(len(order_indices) == 0))
        candidates = []
        for ranked in ranked_candidates:
            idx = ranked["idx"]
            target = ranked["target"]
            to_pose = target["pose"]
            seg, fallback_attempted, fallback_expanded, retries_used, budget_hit = _plan_leg(
                cur_pose,
                to_pose,
                all_obs_xy,
                cfg,
                hx,
                hy,
                deadline=deadline,
            )
            if fallback_attempted:
                attempted_expansions_total += fallback_expanded
                smoothing_retries_used += retries_used
            if budget_hit:
                time_budget_hit = True
            if seg.get("reason") == "time_budget":
                time_budget_hit = True
            if not seg.get("success"):
                continue
            if not bool(getattr(cfg, "allow_reverse_turn", True)) and _segment_has_backward_turn(seg):
                continue
            components = _leg_effective_components(seg, cfg)
            candidates.append(
                {
                    "idx": idx,
                    "tier": ranked["tier"],
                    "rank_key": ranked["key"],
                    "seg": seg,
                    "components": components,
                    "target": target,
                }
            )

        if not candidates:
            break

        if bool(getattr(cfg, "start_straight_bias", False)) and len(order_indices) == 0:
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
        selected.append(picked["target"])
        segments.append(seg)
        total_cost += seg.get("cost", 0.0)
        remaining.remove(idx)
        cur_pose = picked["target"]["pose"]

    reached_all = len(remaining) == 0
    skipped_no_view_state_ids = []
    skipped_unreachable_ids = []
    if remaining:
        final_reference_x = cur_pose[0]
        for idx in sorted(remaining):
            ob = obs_all[idx]
            obstacle_id = ob.get("id", idx)
            cands = generate_view_states(ob, cfg, hx, hy, reference_x_m=final_reference_x)
            _record_bias_debug(idx, bool(cands))
            if not cands:
                skipped_no_view_state_ids.append(obstacle_id)
            else:
                skipped_unreachable_ids.append(obstacle_id)

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

    for chosen, seg in zip(selected, segments):
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
    vertical_bias_bands_m = {
        "low": float(getattr(cfg, "capture_vertical_bias_low_m", 0.0)),
        "mid": float(getattr(cfg, "capture_vertical_bias_mid_m", 0.0)),
        "high": float(getattr(cfg, "capture_vertical_bias_high_m", 0.0)),
    }
    horizontal_bias_bands_m = {
        "low": float(getattr(cfg, "capture_horizontal_bias_low_m", 0.0)),
        "mid": float(getattr(cfg, "capture_horizontal_bias_mid_m", 0.0)),
        "high": float(getattr(cfg, "capture_horizontal_bias_high_m", 0.0)),
    }
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
            "allow_reverse_turn": bool(getattr(cfg, "allow_reverse_turn", True)),
            "horizontal_bias_mode": "current_x_distance",
            "horizontal_bias_reference": "per_leg_current_x",
            "vertical_bias_bands_m": vertical_bias_bands_m,
            "vertical_bias_band_splits_ratio": {
                "low_max": _VERTICAL_BIAS_LOW_MAX_RATIO,
                "mid_max": _VERTICAL_BIAS_MID_MAX_RATIO,
            },
            "horizontal_bias_bands_m": horizontal_bias_bands_m,
            "horizontal_bias_band_splits_ratio": {
                "low_max": _HORIZONTAL_BIAS_LOW_MAX_RATIO,
                "mid_max": _HORIZONTAL_BIAS_MID_MAX_RATIO,
            },
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
            "candidate_count": total_obstacles,
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
                obs_all[idx].get("id", idx) for idx in range(total_obstacles) if idx not in visited_indices
            ],
            "skipped_no_view_state_ids": skipped_no_view_state_ids,
            "skipped_unreachable_ids": skipped_unreachable_ids,
            "skipped_obstacle_ids": skipped_obstacle_ids,
            "obstacle_vertical_bias": [
                obstacle_vertical_bias_by_index[idx] for idx in sorted(obstacle_vertical_bias_by_index.keys())
            ],
            "obstacle_horizontal_bias": [
                obstacle_horizontal_bias_by_index[idx]
                for idx in sorted(obstacle_horizontal_bias_by_index.keys())
            ],
            "leg_effective_components": leg_effective_components,
            "effective_cost_total": effective_cost_total,
        },
    }
