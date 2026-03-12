from dataclasses import dataclass
import math


@dataclass
class PlannerV2Config:
    # Map model (meters)
    map_width_cells: int = 20
    map_height_cells: int = 20
    cell_size: float = 0.10

    # Robot footprint and safety
    robot_L: float = 0.20
    robot_W: float = 0.20
    margin: float = 0.01

    # Obstacles are axis-aligned boxes of this side length (meters)
    obs_size: float = 0.10

    # Hybrid A* discretization and kinematics
    res_xy: float = 0.10
    n_theta: int = 32
    r_min: float = 0.35
    r_min_front: float | None = 0.26
    r_min_back: float | None = 0.36
    primitive_len: float = 0.08
    substep_len: float = 0.02

    # Motion model
    reverse_enabled: bool = True

    # Costs
    w_turn: float = 0.06
    w_reverse: float = 0.08
    w_switch: float = 0.10
    w_steer_switch: float = 0.10
    w_clearance: float = 0.15
    w_heading: float = 0.15

    # Goal tolerance
    pos_tol: float = 0.03
    theta_tol: float = math.radians(10)

    # Search limits
    max_expansions: int = 120000
    analytic_expand_every: int = 20

    # Planner mode and sequencing
    planner_mode: str = "dubins_fallback"
    sequence_mode: str = "greedy_nearest"
    smooth_mode: str = "max"
    planning_time_budget_s: float = 2.0
    hybrid_retry_levels: int = 2
    min_turn_run_strict: int = 2
    min_turn_run_relaxed: int = 1
    connector_order: str = "dubins_local_rs_hybrid"

    # Local bridge (bounded SE(2) search + Dubins-to-goal)
    local_bridge_enabled: bool = True
    local_bridge_step_m: float = 0.10
    local_bridge_radius_m: float = 0.60
    local_bridge_heading_bins: int = 16
    local_bridge_max_nodes: int = 300
    local_bridge_allow_reverse: bool = True

    # Reeds-Shepp connector controls
    rs_enabled: bool = True
    rs_max_cusps: int = 2
    rs_allow_ccc: bool = True

    # Single capture pose offset in grid cells (per obstacle face direction)
    capture_offset_cells: float = 2.0
    # Optional exact face standoff control (meters). When > 0, this overrides
    # capture_offset_cells by computing the center offset from obstacle face.
    capture_face_standoff_m: float = 0.3
    # Optional sensor offset from robot center toward forward direction (meters).
    # When <= 0, robot_L / 2 is used (sensor at front edge).
    sensor_forward_offset_m: float = 0.10
    # Vertical bias to shift camera aim point upward on obstacle (meters).
    # Positive values make robot face higher/more central portion of obstacle.
    capture_vertical_bias_m: float = 0.05

    # Legacy multi-candidate view-state knobs (kept for backward compatibility)
    view_offsets: tuple = (0.30, 0.40)
    view_lateral_offsets: tuple = (-0.05, 0.0, 0.05)

    def min_center_x(self) -> float:
        return self.cell_size

    def min_center_y(self) -> float:
        return self.cell_size

    def max_center_x(self) -> float:
        return (self.map_width_cells - 2) * self.cell_size

    def max_center_y(self) -> float:
        return (self.map_height_cells - 2) * self.cell_size

    def turn_radius_forward(self) -> float:
        if self.r_min_front is not None and self.r_min_front > 0.0:
            return float(self.r_min_front)
        return float(self.r_min)

    def turn_radius_reverse(self) -> float:
        if self.r_min_back is not None and self.r_min_back > 0.0:
            return float(self.r_min_back)
        return float(self.r_min)
