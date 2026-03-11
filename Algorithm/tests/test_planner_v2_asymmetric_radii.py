import math

from planner_v2.api import _merge_commands
from planner_v2.config import PlannerV2Config
from planner_v2.dubins import shortest_dubins_path
from planner_v2.primitives import rollout_primitive
from planner_v2.reeds_shepp_core import solve_reeds_shepp


def _mk_cfg(r_left=0.24, r_right=0.36):
    return PlannerV2Config(
        r_min=min(r_left, r_right),
        r_min_left=r_left,
        r_min_right=r_right,
        primitive_len=0.10,
        substep_len=0.02,
    )


def test_primitives_use_side_specific_curvature():
    cfg = _mk_cfg(r_left=0.20, r_right=0.40)

    fl = rollout_primitive(0.0, 0.0, 0.0, "FL", cfg)
    fr = rollout_primitive(0.0, 0.0, 0.0, "FR", cfg)

    expected_left = cfg.primitive_len / cfg.turn_radius("L")
    expected_right = -cfg.primitive_len / cfg.turn_radius("R")
    assert abs(fl.end_pose[2] - expected_left) < 2e-2
    assert abs(fr.end_pose[2] - expected_right) < 2e-2


def test_turn_merge_uses_side_specific_turn_degrees():
    cfg = _mk_cfg(r_left=0.20, r_right=0.40)
    commands = _merge_commands(["FL", "FR", "RL", "RR"], cfg)

    left_deg = int(round(math.degrees(cfg.primitive_len / cfg.turn_radius("L"))))
    right_deg = int(round(math.degrees(cfg.primitive_len / cfg.turn_radius("R"))))
    assert commands == [
        f"FL{left_deg:03d}",
        f"FR{right_deg:03d}",
        f"BL{left_deg:03d}",
        f"BR{right_deg:03d}",
    ]


def test_asymmetric_dubins_solver_returns_solution():
    start = (0.0, 0.0, 0.0)
    goal = (0.8, 0.3, 0.9)

    solved = shortest_dubins_path(start, goal, r_left=0.22, r_right=0.34)
    assert solved["success"] is True
    assert solved["length"] > 0.0
    assert solved["segments"]


def test_asymmetric_reeds_shepp_solver_returns_solution():
    start = (0.0, 0.0, 0.0)
    goal = (-0.25, 0.30, math.pi / 2.0)

    solved = solve_reeds_shepp(start, goal, r_left=0.24, r_right=0.34, max_cusps=2, allow_ccc=True)
    assert solved["success"] is True
    assert solved["total_length_m"] > 0.0
    assert solved["segments"]


def test_equal_radii_matches_legacy_entrypoint():
    start = (0.0, 0.0, 0.0)
    goal = (0.7, 0.2, 0.8)

    legacy = shortest_dubins_path(start, goal, turn_radius=0.30)
    asymmetric = shortest_dubins_path(start, goal, r_left=0.30, r_right=0.30)

    assert legacy["success"] is True
    assert asymmetric["success"] is True
    assert abs(legacy["length"] - asymmetric["length"]) < 1e-6


def test_config_fallback_deprecation_signal():
    cfg = PlannerV2Config(r_min=0.29)
    assert cfg.single_r_min_fallback_used() is True
    assert abs(cfg.turn_radius("L") - 0.29) < 1e-9
    assert abs(cfg.turn_radius("R") - 0.29) < 1e-9
