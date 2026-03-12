from dataclasses import dataclass
import math


@dataclass
class PlannerV2Config:
    # Map model (meters)
    map_width_cells: int = 22
    map_height_cells: int = 22
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
    # Gear-based minimum turn radii (meters)
    r_min_front: float | None = 0.26
    r_min_back: float | None = 0.27
    # Optional side-specific minimum turn radii (meters).
    # When unset/invalid, planner falls back to r_min (deprecated compatibility).
    r_min_left: float | None = None
    r_min_right: float | None = None
    primitive_len: float = 0.08
    substep_len: float = 0.02

    # Motion model
    reverse_enabled: bool = True

    # Costs
    w_turn: float = 0.08
    w_reverse: float = 0.08
    w_switch: float = 0.10
    w_steer_switch: float = 0.12
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
    # When enabled, first target selection prefers straight-ahead captures from start pose.
    start_straight_bias: bool = False
    # Soft sequencing penalties for favoring shorter/less-turn-heavy legs.
    w_leg_distance_quad: float = 0.08
    w_leg_turn_quad: float = 0.03
    smooth_mode: str = "max"
    planning_time_budget_s: float = 110.0
    hybrid_retry_levels: int = 2
    min_turn_run_strict: int = 2
    min_turn_run_relaxed: int = 1
    connector_order: str = "dubins_local_rs_hybrid"
    # Per-leg planning cap (seconds). 0 disables per-leg slicing.
    leg_time_slice_s: float = 100.0
    # Skip RS connector when remaining leg budget falls below this threshold.
    rs_min_budget_ms: int = 250
    # Use CSC-only Dubins words for speed unless explicitly enabled.
    dubins_allow_ccc: bool = False
    # Asymmetric connector solver runtime knobs.
    asym_solver_max_newton_iters: int = 14
    asym_solver_seed_mode: str = "compact"  # compact|full
    asym_solver_seed_limit: int = 18
    asym_solver_early_exit_residual: float = 0.003
    asym_solver_residual_tol: float = 0.01
    # Hard cap on analytic connector sampling points per segment attempt.
    analytic_sampling_max_points: int = 6000
    # Dock-assist fallback: after direct Dubins fails, try Dubins to a
    # pre-dock pose and a final straight forward segment to standoff.
    dock_assist_enabled: bool = True
    dock_assist_scan_step_m: float = 0.02
    dock_assist_scan_min_extra_m: float = 0.02
    dock_assist_scan_max_extra_m: float = 0.60

    # Local bridge (bounded SE(2) search + Dubins-to-goal)
    local_bridge_enabled: bool = True
    local_bridge_step_m: float = 0.10
    local_bridge_radius_m: float = 0.60
    local_bridge_heading_bins: int = 16
    local_bridge_max_nodes: int = 300
    local_bridge_allow_reverse: bool = True
    # Attempt Dubins-to-goal bridge every N expansions (1 means every node).
    local_bridge_dubins_every: int = 3
    # Skip Dubins bridge attempts when this little budget is left.
    local_bridge_dubins_min_budget_ms: int = 120

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

    def _fallback_radius(self) -> float:
        return max(1e-6, float(self.r_min))

    def _gear_radius(self, gear: int) -> float:
        if gear >= 0:
            if self.r_min_front is not None and self.r_min_front > 0.0:
                return float(self.r_min_front)
            return self._fallback_radius()
        if self.r_min_back is not None and self.r_min_back > 0.0:
            return float(self.r_min_back)
        return self._fallback_radius()

    def turn_radius(self, steer: str, gear: int = 1) -> float:
        s = str(steer).upper()
        base_radius = self._gear_radius(gear)
        if s == "L":
            left = self.r_min_left
            if left is None or left <= 0.0:
                return base_radius
            return float(left)
        if s == "R":
            right = self.r_min_right
            if right is None or right <= 0.0:
                return base_radius
            return float(right)
        return base_radius

    def turn_curvature(self, steer: str, gear: int = 1) -> float:
        s = str(steer).upper()
        if s == "L":
            return 1.0 / self.turn_radius("L", gear)
        if s == "R":
            return -1.0 / self.turn_radius("R", gear)
        return 0.0

    def turn_radius_for_action(self, action: str) -> float:
        if not action or len(action) < 2:
            return self._fallback_radius()
        gear = 1 if str(action)[0].upper() == "F" else -1
        steer = str(action)[1].upper()
        return self.turn_radius(steer, gear)

    def turn_curvature_for_action(self, action: str) -> float:
        if not action or len(action) < 2:
            return 0.0
        gear = 1 if str(action)[0].upper() == "F" else -1
        steer = str(action)[1].upper()
        return self.turn_curvature(steer, gear)

    def min_turn_radius(self) -> float:
        return min(
            self.turn_radius("L", 1),
            self.turn_radius("R", 1),
            self.turn_radius("L", -1),
            self.turn_radius("R", -1),
        )

    def turn_unit_deg(self, steer: str, gear: int = 1) -> float:
        s = str(steer).upper()
        if s not in ("L", "R"):
            return 0.0
        return math.degrees(self.primitive_len / self.turn_radius(s, gear))

    def turn_unit_deg_for_action(self, action: str) -> float:
        if not action or len(action) < 2:
            return 0.0
        gear = 1 if str(action)[0].upper() == "F" else -1
        steer = str(action)[1].upper()
        return self.turn_unit_deg(steer, gear)

    def single_r_min_fallback_used(self) -> bool:
        return (
            self.r_min_front is None
            or self.r_min_front <= 0.0
            or self.r_min_back is None
            or self.r_min_back <= 0.0
        )
