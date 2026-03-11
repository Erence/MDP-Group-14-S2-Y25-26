import math
import time

from planner_v2.config import PlannerV2Config
from planner_v2.dubins import plan_dubins_segment
from planner_v2.dubins_dock import plan_dubins_dock_segment
from planner_v2.geometry import precompute_proj_half_extents


def _mk_cfg():
    return PlannerV2Config(
        r_min=0.30,
        r_min_left=0.30,
        r_min_right=0.30,
        substep_len=0.02,
        dock_assist_enabled=True,
        dock_assist_scan_step_m=0.02,
        dock_assist_scan_min_extra_m=0.02,
        dock_assist_scan_max_extra_m=0.60,
    )


def test_dubins_dock_can_recover_when_direct_dubins_fails():
    cfg = _mk_cfg()
    hx, hy = precompute_proj_half_extents(cfg.robot_L, cfg.robot_W, cfg.margin, cfg.n_theta)

    # Deterministic case found via offline search:
    # direct Dubins fails, but a pre-dock + forward dock chain is feasible.
    start = (0.6268932981412239, 0.7786871636700663, 1.0596762664594648)
    target = (1.0, 1.35, -math.pi / 2.0)
    obstacles_xy = [(1.0, 1.0)]

    direct = plan_dubins_segment(start, target, obstacles_xy, cfg, hx, hy)
    assert direct["success"] is False

    dock = plan_dubins_dock_segment(start, target, obstacles_xy, cfg, hx, hy)
    assert dock["success"] is True
    assert dock["planner"] == "dubins_dock"
    assert dock["commands"]
    assert dock["commands"][-1].startswith("FW")
    assert dock["command_steps"][-1][1] == target
    assert dock["path"][-1] == target


def test_dubins_dock_honors_immediate_deadline():
    cfg = _mk_cfg()
    hx, hy = precompute_proj_half_extents(cfg.robot_L, cfg.robot_W, cfg.margin, cfg.n_theta)
    start = (0.2, 0.2, 0.0)
    target = (0.8, 0.8, math.pi / 2.0)

    out = plan_dubins_dock_segment(
        start,
        target,
        [],
        cfg,
        hx,
        hy,
        deadline=time.monotonic() - 1e-3,
    )
    assert out["success"] is False
    assert out["reason"] == "time_budget"


def test_dubins_dock_disabled_flag():
    cfg = _mk_cfg()
    cfg.dock_assist_enabled = False
    hx, hy = precompute_proj_half_extents(cfg.robot_L, cfg.robot_W, cfg.margin, cfg.n_theta)

    out = plan_dubins_dock_segment((0.2, 0.2, 0.0), (0.8, 0.8, 0.0), [], cfg, hx, hy)
    assert out["success"] is False
    assert out["reason"] == "dubins_dock_disabled"
