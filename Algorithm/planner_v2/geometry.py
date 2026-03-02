import math


def wrap_pi(angle: float) -> float:
    while angle <= -math.pi:
        angle += 2.0 * math.pi
    while angle > math.pi:
        angle -= 2.0 * math.pi
    return angle


def theta_to_bin(theta: float, n_theta: int) -> int:
    theta = theta % (2.0 * math.pi)
    return int(theta / (2.0 * math.pi) * n_theta) % n_theta


def precompute_proj_half_extents(robot_l: float, robot_w: float, margin: float, n_theta: int):
    hx = [0.0] * n_theta
    hy = [0.0] * n_theta
    for k in range(n_theta):
        th = 2.0 * math.pi * k / n_theta
        c = abs(math.cos(th))
        s = abs(math.sin(th))
        hx[k] = c * (robot_l / 2.0 + margin) + s * (robot_w / 2.0 + margin)
        hy[k] = s * (robot_l / 2.0 + margin) + c * (robot_w / 2.0 + margin)
    return hx, hy


def face_to_theta(face_dir: str) -> float:
    f = str(face_dir).upper()
    if f == "N":
        return math.pi / 2.0
    if f == "E":
        return 0.0
    if f == "S":
        return -math.pi / 2.0
    if f == "W":
        return math.pi
    raise ValueError(f"Unsupported face_dir: {face_dir}")


def opposite_face(face_dir: str) -> str:
    f = str(face_dir).upper()
    if f == "N":
        return "S"
    if f == "E":
        return "W"
    if f == "S":
        return "N"
    if f == "W":
        return "E"
    raise ValueError(f"Unsupported face_dir: {face_dir}")
