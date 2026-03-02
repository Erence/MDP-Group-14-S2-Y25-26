import time
import math
from algo.algo import MazeSolver 
from flask import Flask, request, jsonify
from flask_cors import CORS
from model import *
from helper import command_generator
from planner_v2.api import plan_mission_v2
from planner_v2.config import PlannerV2Config
from stm_commands import to_stm_commands

app = Flask(__name__)
CORS(app)
#model = load_model()
model = None

@app.route('/status', methods=['GET'])
def status():
    """
    This is a health check endpoint to check if the server is running
    :return: a json object with a key "result" and value "ok"
    """
    return jsonify({"result": "ok"})


@app.route('/path', methods=['POST'])
def path_finding():
    """
    This is the main endpoint for the path finding algorithm
    :return: a json object with a key "data" and value a dictionary with keys "distance", "path", and "commands"
    """
    # Get the json data from the request
    content = request.json

    # Get the obstacles, big_turn, retrying, robot_x, robot_y, and robot_direction from the json data
    obstacles = content['obstacles']
    # big_turn = int(content['big_turn'])
    retrying = content['retrying']
    robot_x, robot_y = content['robot_x'], content['robot_y']
    robot_direction = int(content['robot_dir'])

    # Initialize MazeSolver object with robot size of 20x20, bottom left corner of robot at (1,1), facing north, and whether to use a big turn or not.
    maze_solver = MazeSolver(20, 20, robot_x, robot_y, robot_direction, big_turn=None)

    # Add each obstacle into the MazeSolver. Each obstacle is defined by its x,y positions, its direction, and its id
    for ob in obstacles:
        maze_solver.add_obstacle(ob['x'], ob['y'], ob['d'], ob['id'])

    start = time.time()
    # Get shortest path
    optimal_path, distance = maze_solver.get_optimal_order_dp(retrying=retrying)
    print(f"Time taken to find shortest path using A* search: {time.time() - start}s")
    print(f"Distance to travel: {distance} units")
    
    # Based on the shortest path, generate commands for the robot
    commands = command_generator(optimal_path, obstacles)
    stm_commands = to_stm_commands(commands)

    # Get the starting location and add it to path_results
    path_results = [optimal_path[0].get_dict()]
    # Process each command individually and append the location the robot should be after executing that command to path_results
    i = 0
    for command in commands:
        if command.startswith("SNAP"):
            continue
        if command.startswith("FIN"):
            continue
        elif command.startswith("FW") or command.startswith("FS"):
            i += int(command[2:]) // 10
        elif command.startswith("BW") or command.startswith("BS"):
            i += int(command[2:]) // 10
        else:
            i += 1
        path_results.append(optimal_path[i].get_dict())
    
    print(f"Commands: {commands}")
    print(f"STM Commands: {stm_commands}")

    return jsonify({
        "data": {
            'distance': distance,
            'path': path_results,
            'commands': commands,
            'stm_commands': stm_commands
        },
        "error": None
    })


@app.route('/path_v2', methods=['POST'])
def path_finding_v2():
    """
    Hybrid A* path planner (v2). Non-breaking: existing /path remains default legacy planner.
    Expected obstacle format supports either:
      - {'x': int, 'y': int, 'face_dir': 'N'|'E'|'S'|'W', 'id': int}
      - {'x': int, 'y': int, 'd': 0|2|4|6, 'id': int}
    Response semantics:
      - data.path is the executed trajectory (not visualization smoothing)
      - data.commands uses merged turn commands (FRddd/FLddd/BRddd/BLddd)
    """
    content = request.json
    obstacles = content.get('obstacles', [])
    robot_x = content.get('robot_x', 1)
    robot_y = content.get('robot_y', 1)
    robot_direction = int(content.get('robot_dir', 0))

    cfg = PlannerV2Config(
        robot_L=float(content.get('robot_L', 0.20)),
        robot_W=float(content.get('robot_W', 0.20)),
        margin=float(content.get('margin', 0.01)),
        res_xy=float(content.get('res_xy', 0.10)),
        n_theta=int(content.get('n_theta', 32)),
        r_min=float(content.get('r_min', 0.18)),
        primitive_len=float(content.get('primitive_len', 0.10)),
        substep_len=float(content.get('substep_len', 0.02)),
        reverse_enabled=bool(content.get('reverse_enabled', True)),
        w_turn=float(content.get('w_turn', 0.06)),
        w_reverse=float(content.get('w_reverse', 0.08)),
        w_switch=float(content.get('w_switch', 0.10)),
        w_steer_switch=float(content.get('w_steer_switch', 0.10)),
        w_clearance=float(content.get('w_clearance', 0.15)),
        pos_tol=float(content.get('pos_tol', 0.05)),
        theta_tol=float(content.get('theta_tol', 0.17453292519943295)),  # 10 degrees
        planner_mode=str(content.get('planner_mode', 'dubins_fallback')),
        sequence_mode=str(content.get('sequence_mode', 'greedy_nearest')),
        smooth_mode=str(content.get('smooth_mode', 'max')),
        planning_time_budget_s=float(content.get('planning_time_budget_s', 2.0)),
        hybrid_retry_levels=int(content.get('hybrid_retry_levels', 2)),
        min_turn_run_strict=int(content.get('min_turn_run_strict', 2)),
        min_turn_run_relaxed=int(content.get('min_turn_run_relaxed', 1)),
        connector_order=str(content.get('connector_order', 'dubins_local_rs_hybrid')),
        local_bridge_enabled=bool(content.get('local_bridge_enabled', True)),
        local_bridge_step_m=float(content.get('local_bridge_step_m', 0.10)),
        local_bridge_radius_m=float(content.get('local_bridge_radius_m', 0.60)),
        local_bridge_heading_bins=int(content.get('local_bridge_heading_bins', 16)),
        local_bridge_max_nodes=int(content.get('local_bridge_max_nodes', 300)),
        local_bridge_allow_reverse=bool(content.get('local_bridge_allow_reverse', True)),
        rs_enabled=bool(content.get('rs_enabled', True)),
        rs_max_cusps=int(content.get('rs_max_cusps', 2)),
        rs_allow_ccc=bool(content.get('rs_allow_ccc', True)),
        capture_offset_cells=float(content.get('capture_offset_cells', 2)),
        capture_face_standoff_m=float(content.get('capture_face_standoff_m', 0.0)),
        sensor_forward_offset_m=float(content.get('sensor_forward_offset_m', 0.0)),
    )

    start = time.time()
    result = plan_mission_v2((robot_x, robot_y, robot_direction), obstacles, cfg)
    elapsed = time.time() - start

    if not result['success']:
        return jsonify({
            "data": {
                "distance": None,
                "path": [],
                "commands": [],
                "stm_commands": [],
                "visit_order": [],
                "selected_view_states": [],
                "debug": result.get("debug", {}),
            },
            "error": "v2_no_path",
        }), 400

    commands = result['commands']
    turn_unit_deg = math.degrees(cfg.primitive_len / cfg.r_min)
    stm_commands = to_stm_commands(commands, turn_unit_deg=turn_unit_deg)

    print(f"Time taken to find v2 path: {elapsed}s")
    print(f"V2 Cost: {result['cost']}")
    print(f"V2 Commands: {commands}")
    print(f"V2 STM Commands: {stm_commands}")

    return jsonify({
        "data": {
            "distance": result['cost'],
            "path": result['path'],
            "commands": commands,
            "stm_commands": stm_commands,
            "visit_order": result['visit_order'],
            "selected_view_states": result['selected_view_states'],
            "debug": result.get('debug', {}),
        },
        "error": None
    })


@app.route('/image', methods=['POST'])
def image_predict():
    """
    This is the main endpoint for the image prediction algorithm
    :return: a json object with a key "result" and value a dictionary with keys "obstacle_id" and "image_id"
    """
    file = request.files['file']
    filename = file.filename
    file.save(os.path.join('uploads', filename))
    # filename format: "<timestamp>_<obstacle_id>_<signal>.jpeg"
    constituents = file.filename.split("_")
    obstacle_id = constituents[1]

    ## Week 8 ## 
    signal = constituents[2].strip(".jpg")
    image_id = predict_image(filename, model, signal)

    ## Week 9 ## 
    # We don't need to pass in the signal anymore
    #image_id = predict_image_week_9(filename,model)

    # Return the obstacle_id and image_id
    result = {
        "obstacle_id": obstacle_id,
        "image_id": image_id
    }
    return jsonify(result)

@app.route('/stitch', methods=['GET'])
def stitch():
    """
    This is the main endpoint for the stitching command. Stitches the images using two different functions, in effect creating two stitches, just for redundancy purposes
    """
    img = stitch_image()
    img.show()
    img2 = stitch_image_own()
    img2.show()
    return jsonify({"result": "ok"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
