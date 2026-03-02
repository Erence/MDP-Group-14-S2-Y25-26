import random
import math

from .collision import collides_swept


def shortcut_smooth(path, obstacles_xy, cfg, hx, hy, trials=80):
    if len(path) < 4:
        return path

    out = list(path)
    rng = random.Random(123)

    def _segment_samples(a, b):
        ax, ay, ath = a
        bx, by, bth = b
        dist = math.hypot(bx - ax, by - ay)
        steps = max(2, int(round(dist / cfg.substep_len)))
        samples = []
        for i in range(1, steps + 1):
            t = i / steps
            x = ax + (bx - ax) * t
            y = ay + (by - ay) * t
            th = ath + (bth - ath) * t
            samples.append((x, y, th))
        return samples

    for _ in range(trials):
        if len(out) < 4:
            break
        i = rng.randint(0, len(out) - 3)
        j = rng.randint(i + 2, len(out) - 1)

        samples = _segment_samples(out[i], out[j])
        if collides_swept(samples, obstacles_xy, cfg, hx, hy):
            continue

        out = out[: i + 1] + out[j:]

    return out
