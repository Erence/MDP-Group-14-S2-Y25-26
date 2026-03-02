import React, { useState, useEffect, useRef } from "react";
import QueryAPI from "./QueryAPI";

const Direction = {
  NORTH: 0,
  EAST: 2,
  SOUTH: 4,
  WEST: 6,
  SKIP: 8,
};

const ObDirection = {
  NORTH: 0,
  EAST: 2,
  SOUTH: 4,
  WEST: 6,
  SKIP: 8,
};

const DirectionToString = {
  0: "Up",
  2: "Right",
  4: "Down",
  6: "Left",
  8: "None",
};

const transformCoord = (x, y) => {
  // Change the coordinate system from (0, 0) at top left to (0, 0) at bottom left
  return { x: 19 - y, y: x };
};

export default function Simulator() {
  const [robotState, setRobotState] = useState({
    x: 1,
    y: 1,
    d: Direction.NORTH,
    s: -1,
  });
  const [startRobot, setStartRobot] = useState({ x: 1, y: 1 });
  const [robotX, setRobotX] = useState(1);
  const [robotY, setRobotY] = useState(1);
  const [robotDir, setRobotDir] = useState(0);
  const [obstacles, setObstacles] = useState([]);
  const [obXInput, setObXInput] = useState(0);
  const [obYInput, setObYInput] = useState(0);
  const [directionInput, setDirectionInput] = useState(ObDirection.NORTH);
  const [isComputing, setIsComputing] = useState(false);
  const [path, setPath] = useState([]);
  const [commands, setCommands] = useState([]);
  const [page, setPage] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);

  // New states for Canvas rendering
  const [cellSize, setCellSize] = useState(20);
  const [isMounted, setIsMounted] = useState(false);
  const canvasRef = useRef(null);

  useEffect(() => {
    setIsMounted(true);
    const updateSize = () => {
      setCellSize(window.innerWidth >= 768 ? 32 : 20);
    };
    updateSize(); // Initial call
    window.addEventListener("resize", updateSize);
    return () => window.removeEventListener("resize", updateSize);
  }, []);

  const generateNewID = () => {
    while (true) {
      let new_id = Math.floor(Math.random() * 10) + 1; // just try to generate an id;
      let ok = true;
      for (const ob of obstacles) {
        if (ob.id === new_id) {
          ok = false;
          break;
        }
      }
      if (ok) {
        return new_id;
      }
    }
  };

  // Drawing the canvas whenever relevant state changes
  useEffect(() => {
    if (!isMounted || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");

    // Dimensions parameters
    const gridOffsetX = 30; // space for y-axis labels
    const gridOffsetY = 0;

    // Set actual canvas resolution
    canvas.width = 20 * cellSize + gridOffsetX;
    canvas.height = 20 * cellSize + 30; // 30px space for x-axis labels

    // Clear previous render
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // 1. Draw Grid Border (Optional, leaving it empty as requested)
    // Draw Axis Labels
    ctx.font = cellSize === 32 ? "14px monospace" : "10px monospace";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";

    for (let i = 0; i < 20; i++) {
      const y = i * cellSize;
      // Y-axis label
      ctx.fillStyle = "#0c4a6e"; // text-sky-900
      ctx.fillText(19 - i, gridOffsetX / 2, y + cellSize / 2);

      // X-axis label
      const x = i * cellSize + gridOffsetX;
      ctx.fillText(i, x + cellSize / 2, 20 * cellSize + 15);
    }

    // Helper: translate simulation grid (0-19) to screen pixels
    const getScreenCoord = (gridX, gridY) => {
      const t = transformCoord(gridX, gridY);
      return {
        x: t.y * cellSize + gridOffsetX,
        y: t.x * cellSize + gridOffsetY,
      };
    };

    // 2. Draw Start Region
    const startP = getScreenCoord(startRobot.x, startRobot.y);
    ctx.fillStyle = "rgba(221, 214, 254, 0.4)"; // light violet
    // The start region is a 3x3 box, the center is startRobot
    ctx.fillRect(startP.x - cellSize, startP.y - cellSize, cellSize * 3, cellSize * 3);

    ctx.fillStyle = "#5b21b6"; // text-violet-800
    ctx.font = "bold 10px sans-serif";
    ctx.fillText("Start", startP.x + cellSize / 2, startP.y + cellSize / 2);

    // 3. Draw Obstacles
    for (const ob of obstacles) {
      const p = getScreenCoord(ob.x, ob.y);
      ctx.fillStyle = "#1d4ed8"; // bg-blue-700
      ctx.fillRect(p.x, p.y, cellSize, cellSize);

      // Draw directional red edge
      ctx.fillStyle = "#ef4444"; // bg-red-500
      const b = 4; // border thickness
      if (ob.d === Direction.NORTH) {
        ctx.fillRect(p.x, p.y, cellSize, b);
      } else if (ob.d === Direction.SOUTH) {
        ctx.fillRect(p.x, p.y + cellSize - b, cellSize, b);
      } else if (ob.d === Direction.EAST) {
        ctx.fillRect(p.x + cellSize - b, p.y, b, cellSize);
      } else if (ob.d === Direction.WEST) {
        ctx.fillRect(p.x, p.y, b, cellSize);
      }
    }

    // 4. Draw Robot (3x3 area)
    let markerGridX = 0;
    let markerGridY = 0;
    if (Number(robotState.d) === Direction.NORTH) markerGridY = 1;
    else if (Number(robotState.d) === Direction.EAST) markerGridX = 1;
    else if (Number(robotState.d) === Direction.SOUTH) markerGridY = -1;
    else if (Number(robotState.d) === Direction.WEST) markerGridX = -1;

    for (let i = -1; i <= 1; i++) {
      for (let j = -1; j <= 1; j++) {
        const p = getScreenCoord(robotState.x + i, robotState.y + j);
        const isMarker = i === markerGridX && j === markerGridY;

        if (isMarker) {
          ctx.fillStyle = robotState.s !== -1 ? "#ef4444" : "#fde047"; // red or yellow
        } else {
          ctx.fillStyle = "#16a34a"; // green-600
        }
        ctx.fillRect(p.x, p.y, cellSize, cellSize);

        // Sub-cell border to distinguish the 3x3 parts
        ctx.strokeStyle = "rgba(255, 255, 255, 0.5)"; // white grid on robot
        ctx.lineWidth = 1;
        ctx.strokeRect(p.x, p.y, cellSize, cellSize);
      }
    }

    // 5. Draw Path Route (Smooth Continuous Curve)
    if (path && path.length > 0) {
      ctx.strokeStyle = "#10b981"; // emerald-500
      ctx.lineWidth = 4;
      ctx.lineJoin = "round";
      ctx.lineCap = "round";
      ctx.beginPath();

      const lastIndex = Math.min(page, path.length - 1);
      const points = [];
      for (let i = 0; i <= lastIndex; i++) {
        if (!path[i]) continue;
        const p = getScreenCoord(path[i].x, path[i].y);
        points.push({ x: p.x + cellSize / 2, y: p.y + cellSize / 2, d: path[i].d });
      }

      if (points.length > 0) {
        ctx.moveTo(points[0].x, points[0].y);

        for (let i = 1; i < points.length; i++) {
          const curr = points[i];
          const prev = points[i - 1];

          // Check if direction changed (turning) and it's not the last point
          if (curr.d !== prev.d && i < points.length - 1) {
            const next = points[i + 1];

            // Draw a straight line up to halfway between prev and curr
            const midInX = (prev.x + curr.x) / 2;
            const midInY = (prev.y + curr.y) / 2;
            ctx.lineTo(midInX, midInY);

            // Calculate halfway between curr and next
            const midOutX = (curr.x + next.x) / 2;
            const midOutY = (curr.y + next.y) / 2;

            // Draw a quadratic curve using the corner (curr) as the control point
            ctx.quadraticCurveTo(curr.x, curr.y, midOutX, midOutY);
          } else {
            // For the very last point, or if direction hasn't changed, 
            // but we need to ensure we reach the exact end coordinate if it's the last point.
            // Actually, if we are in a straight line, it's safe to just lineTo the end of the segment.
            // If the PREVIOUS point was a turn, we already drew a curve that ended at midOut.
            // So a subsequent lineTo to curr will just continue the straight line.
            ctx.lineTo(curr.x, curr.y);
          }
        }
        ctx.stroke();
      }
    }

  }, [robotState, startRobot, obstacles, path, page, cellSize, isMounted]);

  const onChangeX = (event) => {
    if (Number.isInteger(Number(event.target.value))) {
      const nb = Number(event.target.value);
      if (0 <= nb && nb < 20) {
        setObXInput(nb);
        return;
      }
    }
    setObXInput(0);
  };

  const onChangeY = (event) => {
    if (Number.isInteger(Number(event.target.value))) {
      const nb = Number(event.target.value);
      if (0 <= nb && nb <= 19) {
        setObYInput(nb);
        return;
      }
    }
    setObYInput(0);
  };

  const onChangeRobotX = (event) => {
    if (Number.isInteger(Number(event.target.value))) {
      const nb = Number(event.target.value);
      if (1 <= nb && nb < 19) {
        setRobotX(nb);
        return;
      }
    }
    setRobotX(1);
  };

  const onChangeRobotY = (event) => {
    if (Number.isInteger(Number(event.target.value))) {
      const nb = Number(event.target.value);
      if (1 <= nb && nb < 19) {
        setRobotY(nb);
        return;
      }
    }
    setRobotY(1);
  };

  const onClickObstacle = () => {
    if (!obXInput && !obYInput) return;
    const newObstacles = [...obstacles];
    newObstacles.push({
      x: obXInput,
      y: obYInput,
      d: directionInput,
      id: generateNewID(),
    });
    setObstacles(newObstacles);
  };

  const onClickRobot = () => {
    setRobotState({ x: robotX, y: robotY, d: robotDir, s: -1 });
    setStartRobot({ x: robotX, y: robotY });
  };

  const onDirectionInputChange = (event) => {
    setDirectionInput(Number(event.target.value));
  };

  const onRobotDirectionInputChange = (event) => {
    setRobotDir(event.target.value);
  };

  const onRemoveObstacle = (ob) => {
    if (path.length > 0 || isComputing) return;
    const newObstacles = [];
    for (const o of obstacles) {
      if (o.x === ob.x && o.y === ob.y) continue;
      newObstacles.push(o);
    }
    setObstacles(newObstacles);
  };

  const compute = () => {
    setIsComputing(true);
    QueryAPI.query(obstacles, robotX, robotY, robotDir, (data, err) => {
      if (data) {
        setPath(data.data.path);
        const commands = [];
        for (let x of data.data.commands) {
          if (x.startsWith("SNAP")) continue;
          commands.push(x);
        }
        setCommands(commands);
      }
      setIsComputing(false);
    });
  };

  const onResetAll = () => {
    setRobotX(1);
    setRobotDir(0);
    setRobotY(1);
    setRobotState({ x: 1, y: 1, d: Direction.NORTH, s: -1 });
    setStartRobot({ x: 1, y: 1 });
    setPath([]);
    setCommands([]);
    setPage(0);
    setObstacles([]);
  };

  const onReset = () => {
    setRobotX(1);
    setRobotDir(0);
    setRobotY(1);
    setRobotState({ x: 1, y: 1, d: Direction.NORTH, s: -1 });
    setStartRobot({ x: 1, y: 1 });
    setPath([]);
    setCommands([]);
    setPage(0);
  };

  useEffect(() => {
    if (page >= path.length) return;
    setRobotState(path[page]);
  }, [page, path]);

  useEffect(() => {
    if (!isPlaying || path.length === 0) return;
    if (page >= path.length - 1) {
      setIsPlaying(false);
      return;
    }
    const timer = setTimeout(() => {
      setPage((prev) => Math.min(prev + 1, path.length - 1));
    }, 350);

    return () => clearTimeout(timer);
  }, [isPlaying, page, path.length]);

  return (
    <div className="flex flex-col items-center justify-center">
      <div className="flex flex-col items-center text-center bg-[#ddd6fe] rounded-xl shadow-xl mb-4">
        <h2 className="card-title text-black p-2 font-mono">
          ALGORITHM SIMULATOR
        </h2>
      </div>

      <div className="flex flex-col lg:flex-row items-center lg:items-start gap-8">
        <div className="flex flex-col items-center border border-black p-2 rounded-xl bg-white shadow-xl">
          {isMounted ? (
            <canvas ref={canvasRef} className="block" />
          ) : (
            <div className="w-[420px] h-[420px] flex items-center justify-center text-gray-400">Loading Canvas...</div>
          )}
        </div>

        <div className="flex flex-col items-center text-center gap-6">
          <div className="flex flex-col items-center text-center bg-[#ddd6fe] rounded-xl shadow-xl">
            <div className="card-body items-center text-center p-4">
              <h2 className="card-title text-black font-mono">
                Robot Position
              </h2>
              <div className="form-control">
                <label className="input-group input-group-horizontal">
                  <span className="bg-primary p-2">X</span>
                  <input
                    onChange={onChangeRobotX}
                    type="number"
                    placeholder="1"
                    min="1"
                    max="18"
                    className="input input-bordered  text-blue-900 w-20"
                  />
                  <span className="bg-primary p-2">Y</span>
                  <input
                    onChange={onChangeRobotY}
                    type="number"
                    placeholder="1"
                    min="1"
                    max="18"
                    className="input input-bordered  text-blue-900 w-20"
                  />
                  <span className="bg-primary p-2">D</span>
                  <select
                    onChange={onRobotDirectionInputChange}
                    value={robotDir}
                    className="select text-blue-900 py-2 pl-2 pr-6"
                  >
                    <option value={ObDirection.NORTH}>Up</option>
                    <option value={ObDirection.SOUTH}>Down</option>
                    <option value={ObDirection.WEST}>Left</option>
                    <option value={ObDirection.EAST}>Right</option>
                  </select>
                  <button
                    className="btn btn-success p-2"
                    onClick={onClickRobot}
                  >
                    Set
                  </button>
                </label>
              </div>
            </div>
          </div>

          <div className="flex flex-col items-center text-center bg-[#ddd6fe] p-4 rounded-xl shadow-xl">
            <h2 className="card-title text-black pb-2 font-mono">
              Add Obstacles
            </h2>
            <div className="form-control">
              <label className="input-group input-group-horizontal">
                <span className="bg-primary p-2">X</span>
                <input
                  onChange={onChangeX}
                  type="number"
                  placeholder="1"
                  min="0"
                  max="19"
                  className="input input-bordered  text-blue-900 w-20"
                />
                <span className="bg-primary p-2">Y</span>
                <input
                  onChange={onChangeY}
                  type="number"
                  placeholder="1"
                  min="0"
                  max="19"
                  className="input input-bordered  text-blue-900 w-20"
                />
                <span className="bg-primary p-2">D</span>
                <select
                  onChange={onDirectionInputChange}
                  value={directionInput}
                  className="select text-blue-900 py-2 pl-2 pr-6"
                >
                  <option value={ObDirection.NORTH}>Up</option>
                  <option value={ObDirection.SOUTH}>Down</option>
                  <option value={ObDirection.WEST}>Left</option>
                  <option value={ObDirection.EAST}>Right</option>
                  <option value={ObDirection.SKIP}>None</option>
                </select>
                <button
                  className="btn btn-success p-2"
                  onClick={onClickObstacle}
                >
                  Add
                </button>
              </label>
            </div>
          </div>

          <div className="grid grid-cols-4 gap-x-2 gap-y-4 items-center">
            {obstacles.map((ob) => {
              return (
                <div
                  key={ob.id}
                  className="badge flex flex-row text-black bg-sky-100 rounded-xl text-xs md:text-sm h-max border-cyan-500 cursor-pointer hover:bg-sky-200 transition-colors"
                  onClick={() => onRemoveObstacle(ob)}
                >
                  <div className="flex flex-col">
                    <div>X: {ob.x}</div>
                    <div>Y: {ob.y}</div>
                    <div>D: {DirectionToString[ob.d]}</div>
                  </div>
                  <div>
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      fill="none"
                      viewBox="0 0 24 24"
                      className="inline-block w-4 h-4 stroke-current ml-1"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth="2"
                        d="M6 18L18 6M6 6l12 12"
                      ></path>
                    </svg>
                  </div>
                </div>
              );
            })}
          </div>

          <div className="btn-group btn-group-horizontal py-2">
            <button className="btn btn-error" onClick={onResetAll}>
              Reset All
            </button>
            <button className="btn btn-warning" onClick={onReset}>
              Reset Robot
            </button>
            <button className="btn btn-success" onClick={compute}>
              Submit
            </button>
          </div>

          {path.length > 0 && (
            <div className="flex flex-col items-center text-center bg-[#ddd6fe] p-4 rounded-xl shadow-xl my-2 gap-3 w-full max-w-md">
              <div className="flex flex-row items-center gap-4">
                <button
                  className="btn btn-success"
                  onClick={() => setIsPlaying((prev) => !prev)}
                >
                  {isPlaying ? "Pause" : "Play"}
                </button>
                <button
                  className="btn btn-sm btn-outline"
                  onClick={() => {
                    setIsPlaying(false);
                    setPage((prev) => Math.max(prev - 1, 0));
                  }}
                  disabled={page === 0}
                >
                  ←
                </button>
                <span className="text-black">
                  Step: {page + 1} / {path.length}
                </span>
                <button
                  className="btn btn-sm btn-outline"
                  onClick={() => {
                    setIsPlaying(false);
                    setPage((prev) => Math.min(prev + 1, path.length - 1));
                  }}
                  disabled={page === path.length - 1}
                >
                  →
                </button>
                <span className="text-black">{commands[page]}</span>
              </div>
              <input
                type="range"
                min="0"
                max={path.length - 1}
                value={page}
                onChange={(event) => {
                  setIsPlaying(false);
                  setPage(Number(event.target.value));
                }}
                className="range range-primary w-full"
              />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
