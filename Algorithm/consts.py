from enum import Enum


class Direction(int, Enum):
    NORTH = 0
    EAST = 2
    SOUTH = 4
    WEST = 6
    SKIP = 8

    def __int__(self):
        return self.value

    @staticmethod
    def rotation_cost(d1, d2):
        diff = abs(d1 - d2)
        return min(diff, 8 - diff)

MOVE_DIRECTION = [
    (1, 0, Direction.EAST),
    (-1, 0, Direction.WEST),
    (0, 1, Direction.NORTH),
    (0, -1, Direction.SOUTH),
]

TURN_FACTOR = 1

EXPANDED_CELL = 1 # for both agent and obstacles

WIDTH = 20
HEIGHT = 20

ITERATIONS = 2000
TURN_RADIUS = 1

SAFE_COST = 1000 # the cost for the turn in case there is a chance that the robot is touch some obstacle
SCREENSHOT_COST = 50 # the cost for the place where the picture is taken

'''
Main tuning knobs are here:

1)EXPANDED_CELL in consts.py
    Bigger = more clearance from obstacles/walls (safer, may fail more often to find path).
2)SAFE_COST in consts.py + get_safe_cost() in algo.py
    Bigger = planner avoids near-obstacle cells more aggressively.
3)TURN_RADIUS in consts.py (used by turn_wrt_big_turns in algo.py)
    Controls turn displacement geometry (affects turning feasibility near edges/obstacles).
4)big_turn mode in algo.py (set from main.py)
    0 = tighter 3-1 turn, 1 = wider 4-2 turn.
    Wider turns are safer but need more space.
5)ITERATIONS in consts.py
    Search budget for combinations. Higher = better chance to find feasible/better path, slower runtime.
6)retrying request flag in main.py (used in Entity.py)
    True pushes camera/view positions farther from obstacles.
7)Hard safety thresholds in Grid.reachable() in Entity.py
    This is where minimum obstacle/wall distance rules are enforced (including extra turn/pre-turn checks).
8)SCREENSHOT_COST in consts.py
    Not collision safety directly, but affects which viewing positions are chosen (can indirectly change path risk).
Recommended tuning order:

EXPANDED_CELL (1 -> 2 if too close/colliding)
SAFE_COST (1000 -> 2000/3000 if still hugging obstacles)
big_turn=1 if turning collisions happen
ITERATIONS up if “no path” appears too often
'''