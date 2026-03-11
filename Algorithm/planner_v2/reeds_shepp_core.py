from .asymmetric_words import CCC_WORDS, CSC_WORDS, solve_asymmetric_word_set


_GEAR_PATTERNS = (
    (1, 1, 1),
    (-1, -1, -1),
    (1, 1, -1),
    (1, -1, -1),
    (-1, -1, 1),
    (-1, 1, 1),
    (1, -1, 1),
    (-1, 1, -1),
)


def solve_reeds_shepp(
    start_pose,
    goal_pose,
    turn_radius: float | None = None,
    *,
    r_left: float | None = None,
    r_right: float | None = None,
    max_cusps=2,
    allow_ccc=True,
    deadline=None,
    residual_tol: float = 1e-2,
    max_newton_iters: int = 30,
    seed_mode: str = "compact",
    seed_limit: int = 0,
    early_exit_residual: float = 0.0,
):
    if r_left is None or r_right is None:
        if turn_radius is None:
            return {"success": False, "reason": "invalid_turn_radius"}
        r_left = float(turn_radius)
        r_right = float(turn_radius)

    words = list(CSC_WORDS)
    if allow_ccc:
        words.extend(CCC_WORDS)

    solved = solve_asymmetric_word_set(
        start_pose,
        goal_pose,
        float(r_left),
        float(r_right),
        words,
        _GEAR_PATTERNS,
        max_cusps=max(0, int(max_cusps)),
        residual_tol=residual_tol,
        deadline=deadline,
        max_newton_iters=max_newton_iters,
        seed_mode=seed_mode,
        seed_limit=seed_limit,
        early_exit_residual=early_exit_residual,
    )
    if not solved["success"]:
        reason = solved.get("reason", "no_solution")
        if reason == "time_budget":
            return {"success": False, "reason": "time_budget"}
        if reason == "invalid_turn_radius":
            return {"success": False, "reason": "invalid_turn_radius"}
        return {"success": False, "reason": "no_reeds_shepp_solution"}

    return {
        "success": True,
        "segments": solved["segments"],
        "total_length_m": solved["total_length_m"],
        "residual": solved["residual"],
        "cusp_count": solved.get("cusp_count", 0),
        "word": solved.get("word", ""),
        "gear_signature": solved.get("gear_signature", ""),
    }
