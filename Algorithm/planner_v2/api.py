import math
import re
import time

from .collision import collides_swept
from .config import PlannerV2Config
from .geometry import precompute_proj_half_extents, wrap_pi
from .sequencing import plan_sequence
from .smoothing import shortcut_smooth


def _dir_int_to_face(d: int) -> str:
    mapping = {0: "N", 2: "E", 4: "S", 6: "W"}
    if d not in mapping:
        raise ValueError(f"Unsupported direction int: {d}")
    return mapping[d]


def _dir_int_to_theta(d: int) -> float:
    if d == 0:
        return math.pi / 2.0
    if d == 2:
        return 0.0
    if d == 4:
        return -math.pi / 2.0
    if d == 6:
        return math.pi
    raise ValueError(f"Unsupported direction int: {d}")


def _theta_to_dir_int(theta: float) -> int:
    cands = [0.0, math.pi / 2.0, math.pi, -math.pi / 2.0]
    vals = [2, 0, 6, 4]
    diffs = [abs(wrap_pi(theta - c)) for c in cands]
    return vals[diffs.index(min(diffs))]


def _grid_to_m(v: float, cfg: PlannerV2Config) -> float:
    return float(v) * cfg.cell_size


def _m_to_grid(v: float, cfg: PlannerV2Config) -> float:
    return float(v) / cfg.cell_size


def _merge_commands(actions, cfg: PlannerV2Config):
    # Compatibility-oriented command stream. Curved primitives map to merged turn commands.
    cmds = []
    straight_acc = 0.0
    straight_mode = None
    turn_acc = 0.0
    turn_mode = None
    def flush_straight():
        nonlocal straight_acc, straight_mode
        if straight_mode is None or straight_acc <= 0:
            straight_acc = 0.0
            straight_mode = None
            return
        cells = max(1, int(round(straight_acc / cfg.cell_size)))
        val = cells * 10
        if straight_mode == "F":
            cmds.append(f"FW{val}")
        else:
            cmds.append(f"BW{val}")
        straight_acc = 0.0
        straight_mode = None

    def flush_turn():
        nonlocal turn_acc, turn_mode
        if turn_mode is None or turn_acc <= 0:
            turn_acc = 0.0
            turn_mode = None
            return
        deg = max(1, int(round(turn_acc)))
        cmds.append(f"{turn_mode}{deg:03d}")
        turn_acc = 0.0
        turn_mode = None

    for act in actions:
        act_name = act
        straight_len = None

        if isinstance(act, (tuple, list)) and len(act) == 2 and act[0] in ("FS", "RS"):
            act_name = act[0]
            straight_count = int(act[1])
            if straight_count > 0:
                straight_len = straight_count * cfg.primitive_len
        elif act in ("FS", "RS"):
            straight_len = cfg.primitive_len

        if act_name == "FS" and straight_len is not None:
            flush_turn()
            if straight_mode == "R":
                flush_straight()
            straight_mode = "F"
            straight_acc += straight_len
            continue

        if act_name == "RS" and straight_len is not None:
            flush_turn()
            if straight_mode == "F":
                flush_straight()
            straight_mode = "R"
            straight_acc += straight_len
            continue

        flush_straight()
        next_turn_mode = None
        next_turn_unit_deg = 0.0
        if act_name == "FL":
            next_turn_mode = "FL"
            next_turn_unit_deg = cfg.turn_unit_deg_for_action("FL")
        elif act_name == "FR":
            next_turn_mode = "FR"
            next_turn_unit_deg = cfg.turn_unit_deg_for_action("FR")
        elif act_name == "RL":
            next_turn_mode = "BL"
            next_turn_unit_deg = cfg.turn_unit_deg_for_action("RL")
        elif act_name == "RR":
            next_turn_mode = "BR"
            next_turn_unit_deg = cfg.turn_unit_deg_for_action("RR")

        if next_turn_mode is None:
            flush_turn()
            continue

        if turn_mode is not None and turn_mode != next_turn_mode:
            flush_turn()
        turn_mode = next_turn_mode
        turn_acc += next_turn_unit_deg

    flush_straight()
    flush_turn()
    return cmds


def _merge_commands_with_end_poses(actions, path, cfg: PlannerV2Config):
    cmds = []
    ends = []
    straight_acc = 0.0
    straight_mode = None
    straight_end = None
    turn_acc = 0.0
    turn_mode = None
    turn_end = None
    def _path_end_for_step(step_idx: int):
        if not path:
            return None
        if len(path) == 1:
            return path[0]
        idx = min(step_idx + 1, len(path) - 1)
        return path[idx]

    def flush_straight():
        nonlocal straight_acc, straight_mode, straight_end
        if straight_mode is None or straight_acc <= 0:
            straight_acc = 0.0
            straight_mode = None
            straight_end = None
            return
        cells = max(1, int(round(straight_acc / cfg.cell_size)))
        val = cells * 10
        if straight_mode == "F":
            cmds.append(f"FW{val}")
        else:
            cmds.append(f"BW{val}")
        if straight_end is not None:
            ends.append(straight_end)
        straight_acc = 0.0
        straight_mode = None
        straight_end = None

    def flush_turn():
        nonlocal turn_acc, turn_mode, turn_end
        if turn_mode is None or turn_acc <= 0:
            turn_acc = 0.0
            turn_mode = None
            turn_end = None
            return
        deg = max(1, int(round(turn_acc)))
        cmds.append(f"{turn_mode}{deg:03d}")
        if turn_end is not None:
            ends.append(turn_end)
        turn_acc = 0.0
        turn_mode = None
        turn_end = None

    for step_idx, act in enumerate(actions):
        end_pose = _path_end_for_step(step_idx)
        act_name = act
        straight_len = None

        if isinstance(act, (tuple, list)) and len(act) == 2 and act[0] in ("FS", "RS"):
            act_name = act[0]
            straight_count = int(act[1])
            if straight_count > 0:
                straight_len = straight_count * cfg.primitive_len
        elif act in ("FS", "RS"):
            straight_len = cfg.primitive_len

        if act_name == "FS" and straight_len is not None:
            flush_turn()
            if straight_mode == "R":
                flush_straight()
            straight_mode = "F"
            straight_acc += straight_len
            straight_end = end_pose
            continue

        if act_name == "RS" and straight_len is not None:
            flush_turn()
            if straight_mode == "F":
                flush_straight()
            straight_mode = "R"
            straight_acc += straight_len
            straight_end = end_pose
            continue

        flush_straight()
        next_turn_mode = None
        next_turn_unit_deg = 0.0
        if act_name == "FL":
            next_turn_mode = "FL"
            next_turn_unit_deg = cfg.turn_unit_deg_for_action("FL")
        elif act_name == "FR":
            next_turn_mode = "FR"
            next_turn_unit_deg = cfg.turn_unit_deg_for_action("FR")
        elif act_name == "RL":
            next_turn_mode = "BL"
            next_turn_unit_deg = cfg.turn_unit_deg_for_action("RL")
        elif act_name == "RR":
            next_turn_mode = "BR"
            next_turn_unit_deg = cfg.turn_unit_deg_for_action("RR")

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
    return cmds, ends


_MOTION_RE = re.compile(r"^(FW|BW|FR|FL|BR|BL)(\d+)$")


def _parse_motion_command(cmd: str):
    m = _MOTION_RE.fullmatch(str(cmd))
    if not m:
        return None
    opcode = m.group(1)
    num_str = m.group(2)
    return opcode, int(num_str), len(num_str)


def _format_motion_command(opcode: str, value: int, width_hint: int):
    width = width_hint
    if opcode in ("FR", "FL", "BR", "BL"):
        width = max(width, 3)
    return f"{opcode}{value:0{max(1, width)}d}"


def _normalize_turn_motion_command(cmd: str):
    parsed = _parse_motion_command(cmd)
    if parsed is None:
        return cmd

    opcode, value, width = parsed
    if opcode not in ("FR", "FL", "BR", "BL"):
        return cmd

    if value <= 360:
        return cmd

    normalized = value % 360
    if normalized == 0:
        return None
    return _format_motion_command(opcode, normalized, width)


def _normalize_turn_command_steps(command_steps):
    out = []
    for cmd, end_pose in command_steps:
        normalized_cmd = _normalize_turn_motion_command(cmd)
        if normalized_cmd is None:
            continue
        out.append((normalized_cmd, end_pose))
    return out


def _simplify_command_steps_safe(command_steps):
    out = []
    for cmd, end_pose in command_steps:
        parsed = _parse_motion_command(cmd)
        if parsed is None:
            out.append((cmd, end_pose))
            continue

        opcode, value, width = parsed
        if value < 1:
            continue

        if out:
            prev_cmd, _ = out[-1]
            prev_parsed = _parse_motion_command(prev_cmd)
            if prev_parsed is not None and prev_parsed[0] == opcode:
                prev_width = prev_parsed[2]
                merged_val = prev_parsed[1] + value
                out[-1] = (_format_motion_command(opcode, merged_val, max(prev_width, width)), end_pose)
                continue

        out.append((_format_motion_command(opcode, value, width), end_pose))
    return out


def _micro_tweak_score_commands(commands):
    score = 0
    prev = None
    for cmd in commands:
        parsed = _parse_motion_command(cmd)
        if parsed is None:
            continue
        opcode = parsed[0]
        if opcode in ("FR", "BR"):
            side = "R"
        elif opcode in ("FL", "BL"):
            side = "L"
        else:
            continue
        if prev in ("L", "R") and side != prev:
            score += 1
        prev = side
    return score


def _build_obstacles_m(obstacles, cfg: PlannerV2Config):
    out = []
    for ob in obstacles:
        face = ob.get("face_dir")
        if face is None and "d" in ob:
            face = _dir_int_to_face(int(ob["d"]))
        out.append(
            {
                "id": ob.get("id"),
                "x_m": _grid_to_m(ob["x"], cfg),
                "y_m": _grid_to_m(ob["y"], cfg),
                "face_dir": face,
                "raw": ob,
            }
        )
    return out


def plan_mission_v2(start, obstacles, cfg: PlannerV2Config | None = None):
    if cfg is None:
        cfg = PlannerV2Config()

    sx, sy, sdir = start
    start_pose = (_grid_to_m(sx, cfg), _grid_to_m(sy, cfg), _dir_int_to_theta(int(sdir)))

    obstacles_m = _build_obstacles_m(obstacles, cfg)
    missing_face = [ob for ob in obstacles_m if not ob.get("face_dir")]
    if missing_face:
        return {
            "success": False,
            "path": [],
            "commands": [],
            "visit_order": [],
            "selected_view_states": [],
            "cost": float("inf"),
            "debug": {"reason": "missing_face_dir", "count": len(missing_face)},
        }

    hx, hy = precompute_proj_half_extents(cfg.robot_L, cfg.robot_W, cfg.margin, cfg.n_theta)

    budget = float(getattr(cfg, "planning_time_budget_s", 0.0))
    deadline = None
    if budget > 0:
        deadline = time.monotonic() + budget

    seq = plan_sequence(start_pose, obstacles_m, cfg, hx, hy, deadline=deadline)
    if not seq["success"]:
        debug_info = dict(seq.get("debug", {}))
        if "reason" not in debug_info:
            debug_info["reason"] = seq.get("reason", "no_sequence_path")
        return {
            "success": False,
            "path": [],
            "commands": [],
            "visit_order": [],
            "selected_view_states": [],
            "cost": float("inf"),
            "debug": debug_info,
        }

    full_path = [start_pose]
    for seg in seq["segments"]:
        if len(seg["path"]) > 1:
            full_path.extend(seg["path"][1:])

    # Optional shortcut smoothing.
    obs_xy = [(ob["x_m"], ob["y_m"]) for ob in obstacles_m]
    smooth_path = shortcut_smooth(full_path, obs_xy, cfg, hx, hy)
    if collides_swept(smooth_path, obs_xy, cfg, hx, hy):
        smooth_path = full_path

    commands = []
    micro_tweak_scores = []
    path_results = [
        {
            "x": round(_m_to_grid(start_pose[0], cfg), 3),
            "y": round(_m_to_grid(start_pose[1], cfg), 3),
            "d": _theta_to_dir_int(start_pose[2]),
            "s": -1,
        }
    ]

    def append_path_pose(pose):
        path_results.append(
            {
                "x": round(_m_to_grid(pose[0], cfg), 3),
                "y": round(_m_to_grid(pose[1], cfg), 3),
                "d": _theta_to_dir_int(pose[2]),
                "s": -1,
            }
        )

    smooth_mode = str(getattr(cfg, "smooth_mode", "max") or "max").strip().lower()
    for seg, selected in zip(seq["segments"], seq["selected"]):
        leg_steps = []
        planner_name = seg.get("planner")

        if planner_name in ("dubins", "dubins_dock", "reeds_shepp", "local_bridge"):
            seg_steps = seg.get("command_steps", [])
            if seg_steps:
                leg_steps = list(seg_steps)
            else:
                seg_cmds = seg.get("commands", [])
                seg_path = seg.get("path", [])
                if seg_cmds and seg_path:
                    end_pose = seg_path[-1]
                    leg_steps = [(cmd, end_pose) for cmd in seg_cmds]
        else:
            seg_cmds, seg_ends = _merge_commands_with_end_poses(seg.get("actions", []), seg.get("path", []), cfg)
            leg_steps = list(zip(seg_cmds, seg_ends))

        if smooth_mode in ("balanced", "max"):
            leg_steps = _simplify_command_steps_safe(leg_steps)
        leg_steps = _normalize_turn_command_steps(leg_steps)

        leg_commands = [cmd for cmd, _ in leg_steps]
        micro_tweak_scores.append(_micro_tweak_score_commands(leg_commands))
        commands.extend(leg_commands)
        for _, pose in leg_steps:
            append_path_pose(pose)

        obstacle_id = selected.get("obstacle_id")
        if obstacle_id is not None:
            commands.append(f"SNAP{obstacle_id}_C")
    if not commands or commands[-1] != "FIN":
        commands.append("FIN")

    selected_vs = []
    for s in seq["selected"]:
        x, y, th = s["pose"]
        selected_vs.append(
            {
                "obstacle_index": s["obstacle_index"],
                "obstacle_id": s["obstacle_id"],
                "x": round(_m_to_grid(x, cfg), 3),
                "y": round(_m_to_grid(y, cfg), 3),
                "d": _theta_to_dir_int(th),
            }
        )

    debug_info = dict(seq.get("debug", {}))
    debug_info["turn_radii_m"] = {
        "front": cfg.r_min_front if cfg.r_min_front and cfg.r_min_front > 0 else cfg.r_min,
        "back": cfg.r_min_back if cfg.r_min_back and cfg.r_min_back > 0 else cfg.r_min,
        "left_override": cfg.r_min_left,
        "right_override": cfg.r_min_right,
    }
    debug_info["single_r_min_fallback_used"] = cfg.single_r_min_fallback_used()
    debug_info["micro_tweak_score_per_leg"] = micro_tweak_scores
    debug_info["smoothing"] = {
        "applied": len(smooth_path) != len(full_path),
        "executed_points": len(full_path),
        "smoothed_points": len(smooth_path),
    }

    return {
        "success": True,
        "path": path_results,
        "commands": commands,
        "visit_order": seq["visit_order"],
        "selected_view_states": selected_vs,
        "cost": seq["cost"],
        "debug": debug_info,
    }
