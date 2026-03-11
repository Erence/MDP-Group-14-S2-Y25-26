import math
import time

from .geometry import wrap_pi


CSC_WORDS = (
    ("L", "S", "L"),
    ("L", "S", "R"),
    ("R", "S", "L"),
    ("R", "S", "R"),
)

CCC_WORDS = (
    ("L", "R", "L"),
    ("R", "L", "R"),
)


def _transform_to_local(start_pose, goal_pose):
    sx, sy, sth = start_pose
    gx, gy, gth = goal_pose
    dx = gx - sx
    dy = gy - sy
    cs = math.cos(sth)
    sn = math.sin(sth)
    x = cs * dx + sn * dy
    y = -sn * dx + cs * dy
    phi = wrap_pi(gth - sth)
    return x, y, phi


def _segment_kappa(seg_type: str, r_left: float, r_right: float):
    if seg_type == "L":
        return 1.0 / r_left
    if seg_type == "R":
        return -1.0 / r_right
    return 0.0


def _integrate_word(word, gears, lengths_m, r_left: float, r_right: float):
    x = 0.0
    y = 0.0
    th = 0.0
    for seg_type, gear, seg_len in zip(word, gears, lengths_m):
        l = max(0.0, float(seg_len))
        if l <= 0.0:
            continue
        ds = float(1 if int(gear) >= 0 else -1) * l
        if seg_type == "S":
            x += ds * math.cos(th)
            y += ds * math.sin(th)
            continue

        kappa = _segment_kappa(seg_type, r_left, r_right)
        nth = th + kappa * ds
        x += (math.sin(nth) - math.sin(th)) / kappa
        y += (-math.cos(nth) + math.cos(th)) / kappa
        th = nth
    return x, y, wrap_pi(th)


def _error_vec(word, gears, lengths_m, target, r_left: float, r_right: float):
    tx, ty, tth = target
    x, y, th = _integrate_word(word, gears, lengths_m, r_left, r_right)
    return [x - tx, y - ty, wrap_pi(th - tth)]


def _weighted_norm(err):
    return math.sqrt(err[0] * err[0] + err[1] * err[1] + err[2] * err[2])


def _solve_linear_3x3(a, b):
    m = [
        [float(a[0][0]), float(a[0][1]), float(a[0][2]), float(b[0])],
        [float(a[1][0]), float(a[1][1]), float(a[1][2]), float(b[1])],
        [float(a[2][0]), float(a[2][1]), float(a[2][2]), float(b[2])],
    ]

    for col in range(3):
        pivot = col
        for row in range(col + 1, 3):
            if abs(m[row][col]) > abs(m[pivot][col]):
                pivot = row
        if abs(m[pivot][col]) < 1e-10:
            return None
        if pivot != col:
            m[col], m[pivot] = m[pivot], m[col]

        inv = 1.0 / m[col][col]
        for k in range(col, 4):
            m[col][k] *= inv

        for row in range(3):
            if row == col:
                continue
            factor = m[row][col]
            if abs(factor) <= 1e-14:
                continue
            for k in range(col, 4):
                m[row][k] -= factor * m[col][k]

    return [m[0][3], m[1][3], m[2][3]]


def _finite_diff_jacobian(word, gears, lengths_m, target, base_err, r_left: float, r_right: float):
    eps = 1e-4
    j = [[0.0, 0.0, 0.0] for _ in range(3)]
    for col in range(3):
        nudged = list(lengths_m)
        nudged[col] = max(0.0, nudged[col] + eps)
        e2 = _error_vec(word, gears, nudged, target, r_left, r_right)
        j[0][col] = (e2[0] - base_err[0]) / eps
        j[1][col] = (e2[1] - base_err[1]) / eps
        j[2][col] = wrap_pi(e2[2] - base_err[2]) / eps
    return j


def _seed_vectors(
    x: float,
    y: float,
    phi: float,
    r_left: float,
    r_right: float,
    *,
    seed_mode: str = "compact",
    seed_limit: int = 0,
):
    d = math.hypot(x, y)
    r_ref = min(r_left, r_right)
    a = abs(phi) * r_ref
    vmax = max(0.20, d, a, math.pi * r_ref / 2.0)
    mode = str(seed_mode or "compact").strip().lower()
    seeds = []
    if mode == "full":
        vals = [0.0, 0.05, 0.10, 0.20, 0.40, d, a, vmax]
        seeds.extend(
            [
                (max(0.05, d), max(0.05, d), max(0.05, a)),
                (max(0.05, a), max(0.05, d), max(0.05, a)),
                (max(0.05, d / 2.0), max(0.05, d / 2.0), max(0.05, a / 2.0)),
                (0.20, 0.20, 0.20),
                (0.10, 0.10, 0.10),
            ]
        )
        for v0 in vals[:6]:
            for v1 in vals[2:8]:
                v2 = max(0.0, a + 0.5 * (v0 - v1))
                seeds.append((max(0.0, v0), max(0.0, v1), max(0.0, v2)))
    else:
        vals = [0.0, 0.08, 0.16, max(0.16, min(d, vmax)), max(0.16, min(a, vmax))]
        seeds.extend(
            [
                (max(0.05, d), max(0.05, d), max(0.05, a)),
                (max(0.05, a), max(0.05, d), max(0.05, a)),
                (0.16, 0.16, 0.16),
                (0.08, 0.08, 0.08),
            ]
        )
        for v0 in vals[:4]:
            for v1 in vals[1:5]:
                v2 = max(0.0, a + 0.5 * (v0 - v1))
                seeds.append((max(0.0, v0), max(0.0, v1), max(0.0, v2)))

    uniq = []
    seen = set()
    for s in seeds:
        key = (round(s[0], 3), round(s[1], 3), round(s[2], 3))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(s)
    if seed_limit and seed_limit > 0:
        return uniq[: int(seed_limit)]
    return uniq


def _count_cusps(gears, lengths_m):
    filtered = []
    for g, l in zip(gears, lengths_m):
        if l > 1e-3:
            filtered.append(g)
    if len(filtered) <= 1:
        return 0
    cusps = 0
    for i in range(1, len(filtered)):
        if filtered[i] != filtered[i - 1]:
            cusps += 1
    return cusps


def _solve_word(
    word,
    gears,
    target,
    seeds,
    r_left: float,
    r_right: float,
    deadline=None,
    *,
    max_newton_iters: int = 30,
    early_exit_residual: float = 0.0,
):
    best = None
    iters = max(1, int(max_newton_iters))
    early_eps = max(0.0, float(early_exit_residual))
    for seed in seeds:
        if deadline is not None and time.monotonic() >= deadline:
            break

        lengths_m = [max(0.0, seed[0]), max(0.0, seed[1]), max(0.0, seed[2])]
        for _ in range(iters):
            if deadline is not None and time.monotonic() >= deadline:
                break
            err = _error_vec(word, gears, lengths_m, target, r_left, r_right)
            err_n = _weighted_norm(err)
            if err_n < 1e-4:
                break

            jac = _finite_diff_jacobian(word, gears, lengths_m, target, err, r_left, r_right)
            delta = _solve_linear_3x3(jac, [-err[0], -err[1], -err[2]])
            if delta is None:
                break

            improved = False
            alpha = 1.0
            while alpha >= 0.0625:
                cand = [
                    max(0.0, lengths_m[0] + alpha * delta[0]),
                    max(0.0, lengths_m[1] + alpha * delta[1]),
                    max(0.0, lengths_m[2] + alpha * delta[2]),
                ]
                c_err = _error_vec(word, gears, cand, target, r_left, r_right)
                if _weighted_norm(c_err) < err_n:
                    lengths_m = cand
                    improved = True
                    break
                alpha *= 0.5
            if not improved:
                break

        final_norm = _weighted_norm(_error_vec(word, gears, lengths_m, target, r_left, r_right))
        if best is None or final_norm < best["residual"]:
            best = {
                "lengths_m": lengths_m,
                "residual": final_norm,
            }
        if best is not None and early_eps > 0.0 and best["residual"] <= early_eps:
            break

    return best


def solve_asymmetric_word_set(
    start_pose,
    goal_pose,
    r_left: float,
    r_right: float,
    words,
    gear_patterns,
    *,
    max_cusps=2,
    residual_tol=1e-2,
    deadline=None,
    max_newton_iters=30,
    seed_mode="compact",
    seed_limit=0,
    early_exit_residual=0.0,
):
    if r_left <= 0.0 or r_right <= 0.0:
        return {"success": False, "reason": "invalid_turn_radius"}

    tx, ty, tphi = _transform_to_local(start_pose, goal_pose)
    target = (tx, ty, tphi)
    max_newton_iters = max(1, int(max_newton_iters))
    seed_limit = max(0, int(seed_limit))
    residual_tol = max(1e-4, float(residual_tol))
    early_exit_residual = max(0.0, float(early_exit_residual))
    seeds = _seed_vectors(
        tx,
        ty,
        tphi,
        r_left,
        r_right,
        seed_mode=seed_mode,
        seed_limit=seed_limit,
    )

    best = None
    time_budget_hit = False
    for word in words:
        if deadline is not None and time.monotonic() >= deadline:
            time_budget_hit = True
            break
        for gears in gear_patterns:
            if deadline is not None and time.monotonic() >= deadline:
                time_budget_hit = True
                break
            rough_cusps = 0
            for i in range(1, len(gears)):
                if gears[i] != gears[i - 1]:
                    rough_cusps += 1
            if rough_cusps > max_cusps:
                continue

            solved = _solve_word(
                word,
                gears,
                target,
                seeds,
                r_left,
                r_right,
                deadline=deadline,
                max_newton_iters=max_newton_iters,
                early_exit_residual=early_exit_residual,
            )
            if solved is None:
                continue
            lengths_m = solved["lengths_m"]
            residual = solved["residual"]
            if not math.isfinite(residual) or residual > residual_tol:
                continue

            cusps = _count_cusps(gears, lengths_m)
            if cusps > max_cusps:
                continue

            segs = []
            total = 0.0
            nonzero_types = []
            nonzero_gears = []
            for seg_type, gear, seg_len in zip(word, gears, lengths_m):
                l = float(seg_len)
                if l <= 1e-6:
                    continue
                if not math.isfinite(l):
                    continue
                total += l
                segs.append(
                    {
                        "type": seg_type,
                        "gear": int(1 if gear >= 0 else -1),
                        "length_m": l,
                    }
                )
                nonzero_types.append(seg_type)
                nonzero_gears.append("F" if gear > 0 else "B")

            if not segs:
                continue

            cand = {
                "segments": segs,
                "total_length_m": total,
                "residual": residual,
                "cusp_count": cusps,
                "word": "".join(nonzero_types),
                "gear_signature": ",".join(nonzero_gears),
            }
            if best is None or cand["total_length_m"] < best["total_length_m"]:
                best = cand
            if (
                best is not None
                and early_exit_residual > 0.0
                and residual <= early_exit_residual
                and best["total_length_m"] <= total
            ):
                return {"success": True, **best}

    if best is None:
        return {"success": False, "reason": "time_budget" if time_budget_hit else "no_solution"}

    return {"success": True, **best}
