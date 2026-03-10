from dataclasses import replace
import time

from .dubins import plan_dubins_segment
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


def _base_debug(cfg, candidate_count: int):
    return {
        "planner_mode": getattr(cfg, "planner_mode", "dubins_fallback"),
        "sequence_mode": getattr(cfg, "sequence_mode", "greedy_nearest"),
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

    if deadline is not None and time.monotonic() >= deadline:
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
        dubins_seg = plan_dubins_segment(from_pose, to_pose, obstacles_xy, cfg, hx, hy)
        record_attempt("dubins", dubins_seg)
        if dubins_seg.get("success"):
            return _finalize_seg(dubins_seg, attempts, start_ts), False, 0, 0, False
        if dubins_seg.get("reason") == "time_budget":
            return _finalize_seg(dubins_seg, attempts, start_ts), False, 0, 0, True

        local_seg = plan_local_bridge_segment(from_pose, to_pose, obstacles_xy, cfg, hx, hy, deadline=deadline)
        record_attempt("local_bridge", local_seg)
        if local_seg.get("success"):
            return _finalize_seg(local_seg, attempts, start_ts), False, 0, 0, False
        if local_seg.get("reason") == "time_budget":
            return _finalize_seg(local_seg, attempts, start_ts), False, 0, 0, True

        if bool(getattr(cfg, "rs_enabled", True)) and bool(getattr(cfg, "reverse_enabled", True)):
            rs_seg = plan_reeds_shepp_segment(from_pose, to_pose, obstacles_xy, cfg, hx, hy, deadline=deadline)
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
        if deadline is not None and time.monotonic() >= deadline:
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
            deadline=deadline,
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
    for oi, ob in enumerate(obs_all):
        cands = generate_view_states(ob, cfg, hx, hy)
        if not cands:
            return {
                "success": False,
                "reason": f"no_view_state_for_obstacle_{ob.get('id', 'unknown')}",
                "debug": _base_debug(cfg, len(obs_all)),
            }
        targets.append(
            {
                "obstacle_index": oi,
                "obstacle_id": ob.get("id", oi),
                "pose": cands[0],
            }
        )

    n_obs = len(targets)
    if n_obs == 0:
        return {
            "success": True,
            "visit_order": [],
            "selected": [],
            "segments": [],
            "cost": 0.0,
            "debug": _base_debug(cfg, 0),
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
        ranked = []
        for idx in remaining:
            to_pose = _pose_of(idx)
            dist_sq = (from_pose[0] - to_pose[0]) ** 2 + (from_pose[1] - to_pose[1]) ** 2
            ranked.append((dist_sq, idx))
        ranked.sort(key=lambda item: item[0])
        return [idx for _, idx in ranked]

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
                return False, order_indices, segments, total_cost

            picked = None
            for idx in _ordered_candidates(cur_idx, remaining):
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
                    picked = (idx, seg)
                    break

            if picked is None:
                return False, order_indices, segments, total_cost

            idx, seg = picked
            order_indices.append(idx)
            segments.append(seg)
            total_cost += seg.get("cost", 0.0)
            remaining.remove(idx)
            cur_idx = idx

        return True, order_indices, segments, total_cost

    def _dfs_route(use_cache: bool, active_cfg, active_deadline):
        nonlocal attempted_expansions_total, smoothing_retries_used, time_budget_hit
        memo = {}

        def _leg(cur_idx: int, idx: int):
            nonlocal attempted_expansions_total, smoothing_retries_used, time_budget_hit
            if use_cache:
                return _cached_leg(cur_idx, idx)
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
            return seg

        def _search(cur_idx: int, remaining: tuple):
            nonlocal time_budget_hit
            if not remaining:
                return True, [], [], 0.0
            if active_deadline is not None and time.monotonic() >= active_deadline:
                time_budget_hit = True
                return False, [], [], float("inf")

            state = (cur_idx, remaining)
            if state in memo:
                return memo[state]

            for idx in _ordered_candidates(cur_idx, remaining):
                seg = _leg(cur_idx, idx)
                if seg.get("reason") == "time_budget":
                    time_budget_hit = True
                if not seg.get("success"):
                    continue
                next_remaining = tuple(r for r in remaining if r != idx)
                ok, tail_order, tail_segments, tail_cost = _search(idx, next_remaining)
                if ok:
                    out = (
                        True,
                        [idx] + tail_order,
                        [seg] + tail_segments,
                        seg.get("cost", 0.0) + tail_cost,
                    )
                    memo[state] = out
                    return out

            out = (False, [], [], float("inf"))
            memo[state] = out
            return out

        return _search(-1, tuple(range(n_obs)))

    ok, order_indices, segments, total_cost = _greedy_route(True, cfg, deadline)
    greedy_order_indices = list(order_indices)
    greedy_segments = list(segments)
    greedy_cost = float(total_cost)
    if not ok:
        ok_dfs, dfs_order, dfs_segments, dfs_cost = _dfs_route(True, cfg, deadline)
        if ok_dfs:
            ok = True
            order_indices = dfs_order
            segments = dfs_segments
            total_cost = dfs_cost

    best_effort_returned = False
    partial = False
    partial_reason = None
    if not ok and time_budget_hit and greedy_order_indices:
        # Budget exceeded after planning a feasible prefix; return it as partial.
        ok = True
        order_indices = greedy_order_indices
        segments = greedy_segments
        total_cost = greedy_cost
        partial = True
        partial_reason = "time_budget"

    if not ok:
        dbg = _base_debug(cfg, n_obs)
        dbg["attempted_expansions_total"] = attempted_expansions_total
        dbg["smoothing_retries_used"] = smoothing_retries_used
        dbg["time_budget_hit"] = time_budget_hit
        dbg["best_effort_returned"] = best_effort_returned
        dbg["remaining_obstacle_ids"] = [t["obstacle_id"] for t in targets]
        return {
            "success": False,
            "reason": "no_sequence_path",
            "debug": dbg,
        }

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

        if planner_name == "dubins":
            dubins_used_count += 1
            leg_planners.append("dubins")
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
            "total_obstacles": n_obs,
            "remaining_obstacle_ids": [
                targets[idx]["obstacle_id"] for idx in range(n_obs) if idx not in visited_indices
            ],
        },
    }
