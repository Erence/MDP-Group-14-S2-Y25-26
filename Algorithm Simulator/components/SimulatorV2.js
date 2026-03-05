import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import QueryAPI from "./QueryAPI";
import {
  DEFAULT_TURN_RADIUS_CELLS,
  WORLD_SIZE,
  WORLD_MIN,
  WORLD_MAX,
  CELL_SIZE,
  ROBOT_RADIUS_CELLS,
  activeSegmentAtTimeline,
  advanceTimeline,
  allSamplePoints,
  buildTrajectoryFromCommands,
  createStartPose,
  directionToTheta,
  frameAtTimeline,
  poseAtTimeline,
  toSvgPoint,
  toSvgPolyline,
  visitedSamplePointsAtTimeline,
} from "./simV2/kinematics";

const Direction = {
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

const WORLD_TICKS = Array.from(
  { length: WORLD_SIZE + 1 },
  (_, index) => WORLD_MIN + index * CELL_SIZE
);
const robotBottomLeftToCenter = (x, y) => ({ x: Number(x) + 1, y: Number(y) + 1 });
const DEFAULT_ROBOT_BOTTOM_LEFT = { x: 0, y: 0, d: Direction.NORTH };
const START_ZONE_RADIUS_CELLS = ROBOT_RADIUS_CELLS;
const HEADING_MARKER_FACTOR = 0.85;
const ANCHOR_MARKER_HALF_WIDTH = 0.08;

function classNames(...classes) {
  return classes.filter(Boolean).join(" ");
}

function formatErrorMessage(error) {
  if (!error) {
    return "Path request failed.";
  }
  if (typeof error === "string") {
    return error;
  }
  if (error.message) {
    return error.message;
  }
  return "Path request failed.";
}

function obstacleIndexToWorldCenter(obstacle) {
  return {
    x: Number(obstacle.x) + 0.5,
    y: Number(obstacle.y) + 0.5,
  };
}

function obstacleFaceLine(obstacle) {
  const center = toSvgPoint(obstacleIndexToWorldCenter(obstacle));
  if (obstacle.d === Direction.NORTH) {
    return {
      x1: center.x - 0.5,
      y1: center.y - 0.5,
      x2: center.x + 0.5,
      y2: center.y - 0.5,
    };
  }
  if (obstacle.d === Direction.SOUTH) {
    return {
      x1: center.x - 0.5,
      y1: center.y + 0.5,
      x2: center.x + 0.5,
      y2: center.y + 0.5,
    };
  }
  if (obstacle.d === Direction.EAST) {
    return {
      x1: center.x + 0.5,
      y1: center.y - 0.5,
      x2: center.x + 0.5,
      y2: center.y + 0.5,
    };
  }
  if (obstacle.d === Direction.WEST) {
    return {
      x1: center.x - 0.5,
      y1: center.y - 0.5,
      x2: center.x - 0.5,
      y2: center.y + 0.5,
    };
  }
  return null;
}

function robotBottomLeftFromPose(x, y) {
  return {
    x: x - ROBOT_RADIUS_CELLS,
    y: y - ROBOT_RADIUS_CELLS,
  };
}

export default function SimulatorV2() {
  const defaultCenter = robotBottomLeftToCenter(
    DEFAULT_ROBOT_BOTTOM_LEFT.x,
    DEFAULT_ROBOT_BOTTOM_LEFT.y
  );
  const [robotX, setRobotX] = useState(DEFAULT_ROBOT_BOTTOM_LEFT.x);
  const [robotY, setRobotY] = useState(DEFAULT_ROBOT_BOTTOM_LEFT.y);
  const [robotDir, setRobotDir] = useState(Direction.NORTH);
  const [startRobot, setStartRobot] = useState({
    x: defaultCenter.x,
    y: defaultCenter.y,
    d: Direction.NORTH,
  });
  const [obstacles, setObstacles] = useState([]);
  const [obXInput, setObXInput] = useState(0);
  const [obYInput, setObYInput] = useState(0);
  const [directionInput, setDirectionInput] = useState(Direction.NORTH);
  const [isComputing, setIsComputing] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");
  const [responsePath, setResponsePath] = useState([]);
  const [responseCommands, setResponseCommands] = useState([]);
  const [selectedViewStates, setSelectedViewStates] = useState([]);
  const [trajectory, setTrajectory] = useState(() =>
    buildTrajectoryFromCommands(
      [],
      createStartPose(defaultCenter.x, defaultCenter.y, Direction.NORTH),
      {
        turnRadiusCells: DEFAULT_TURN_RADIUS_CELLS,
      }
    )
  );
  const [timelinePos, setTimelinePos] = useState(0);

  const timelineRef = useRef(0);
  const animationFrameRef = useRef(null);
  const lastTimestampRef = useRef(null);

  const segments = trajectory.segments;
  const segmentCount = segments.length;

  const currentFrame = useMemo(
    () => frameAtTimeline(trajectory, timelinePos),
    [trajectory, timelinePos]
  );
  const activeSegmentIndex = useMemo(
    () => activeSegmentAtTimeline(trajectory, timelinePos),
    [trajectory, timelinePos]
  );
  const robotPose = useMemo(
    () => poseAtTimeline(trajectory, timelinePos),
    [trajectory, timelinePos]
  );
  const fullPathPoints = useMemo(() => allSamplePoints(trajectory), [trajectory]);
  const visitedPathPoints = useMemo(
    () => visitedSamplePointsAtTimeline(trajectory, timelinePos),
    [trajectory, timelinePos]
  );
  const fullPathPolyline = useMemo(
    () => toSvgPolyline(fullPathPoints),
    [fullPathPoints]
  );
  const visitedPathPolyline = useMemo(
    () => toSvgPolyline(visitedPathPoints),
    [visitedPathPoints]
  );

  const activeCommandIndex =
    activeSegmentIndex >= 0 ? segments[activeSegmentIndex]?.commandIndex ?? -1 : -1;
  const currentCommand =
    activeSegmentIndex >= 0 ? segments[activeSegmentIndex]?.command ?? "-" : "-";

  const setTimeline = useCallback(
    (nextValue) => {
      const clamped = Math.min(Math.max(nextValue, 0), segmentCount);
      timelineRef.current = clamped;
      setTimelinePos(clamped);
    },
    [segmentCount]
  );

  const resetTrajectoryState = useCallback(
    (x, y, d) => {
      setIsPlaying(false);
      setTimeline(0);
      setResponsePath([]);
      setResponseCommands([]);
      setSelectedViewStates([]);
      setTrajectory(
        buildTrajectoryFromCommands([], createStartPose(x, y, d), {
          turnRadiusCells: DEFAULT_TURN_RADIUS_CELLS,
        })
      );
    },
    [setTimeline]
  );

  const generateNewID = useCallback(() => {
    if (obstacles.length === 0) {
      return 1;
    }
    const maxId = obstacles.reduce(
      (best, obstacle) => Math.max(best, Number(obstacle.id) || 0),
      0
    );
    return maxId + 1;
  }, [obstacles]);

  const onChangeX = (event) => {
    if (Number.isInteger(Number(event.target.value))) {
      const next = Number(event.target.value);
      if (next >= 0 && next <= 19) {
        setObXInput(next);
        return;
      }
    }
    setObXInput(0);
  };

  const onChangeY = (event) => {
    if (Number.isInteger(Number(event.target.value))) {
      const next = Number(event.target.value);
      if (next >= 0 && next <= 19) {
        setObYInput(next);
        return;
      }
    }
    setObYInput(0);
  };

  const onChangeRobotX = (event) => {
    if (Number.isInteger(Number(event.target.value))) {
      const next = Number(event.target.value);
      if (next >= 0 && next <= 17) {
        setRobotX(next);
        return;
      }
    }
    setRobotX(0);
  };

  const onChangeRobotY = (event) => {
    if (Number.isInteger(Number(event.target.value))) {
      const next = Number(event.target.value);
      if (next >= 0 && next <= 17) {
        setRobotY(next);
        return;
      }
    }
    setRobotY(0);
  };

  const onDirectionInputChange = (event) => {
    setDirectionInput(Number(event.target.value));
  };

  const onRobotDirectionInputChange = (event) => {
    setRobotDir(Number(event.target.value));
  };

  const onClickObstacle = () => {
    if (isComputing) {
      return;
    }
    setObstacles((previous) => [
      ...previous,
      {
        x: obXInput,
        y: obYInput,
        d: directionInput,
        id: generateNewID(),
      },
    ]);
  };

  const onRemoveObstacle = (obstacle) => {
    if (responseCommands.length > 0 || segmentCount > 0 || isComputing) {
      return;
    }
    setObstacles((previous) =>
      previous.filter((candidate) => candidate.id !== obstacle.id)
    );
  };

  const onClickRobot = () => {
    const nextRobotBottomLeft = {
      x: robotX,
      y: robotY,
      d: Number(robotDir),
    };
    const nextRobotCenter = robotBottomLeftToCenter(
      nextRobotBottomLeft.x,
      nextRobotBottomLeft.y
    );
    setStartRobot({ ...nextRobotCenter, d: nextRobotBottomLeft.d });
    setErrorMessage("");
    resetTrajectoryState(nextRobotCenter.x, nextRobotCenter.y, nextRobotBottomLeft.d);
  };

  const compute = () => {
    if (isComputing) {
      return;
    }

    const nextRobotBottomLeft = {
      x: robotX,
      y: robotY,
      d: Number(robotDir),
    };
    const nextRobotCenter = robotBottomLeftToCenter(
      nextRobotBottomLeft.x,
      nextRobotBottomLeft.y
    );

    setStartRobot({ ...nextRobotCenter, d: nextRobotBottomLeft.d });
    setErrorMessage("");
    setIsPlaying(false);
    setIsComputing(true);

    QueryAPI.query(
      obstacles,
      nextRobotBottomLeft.x,
      nextRobotBottomLeft.y,
      nextRobotBottomLeft.d,
      (result) => {
        if (result?.data) {
          const payload = result.data;
          const commands = Array.isArray(payload.commands) ? payload.commands : [];
          const path = Array.isArray(payload.path) ? payload.path : [];
          const views = Array.isArray(payload.selected_view_states)
            ? payload.selected_view_states
            : [];

          const commandPoseOverrides = {};
          let viewIndex = 0;
          let lastMotionCommandIndex = null;
          commands.forEach((command, commandIndex) => {
            const token = String(command ?? "").trim().toUpperCase();
            if (/^(FW|BW|FR|FL|BR|BL)\d+$/.test(token)) {
              lastMotionCommandIndex = commandIndex;
              return;
            }

            if (token.startsWith("SNAP")) {
              const view = views[viewIndex];
              if (view && lastMotionCommandIndex !== null) {
                commandPoseOverrides[lastMotionCommandIndex] = {
                  theta: directionToTheta(Number(view.d)),
                };
              }
              viewIndex += 1;
            }
          });

          setResponseCommands(commands);
          setResponsePath(path);
          setSelectedViewStates(views);
          setTrajectory(
            buildTrajectoryFromCommands(
              commands,
              createStartPose(nextRobotCenter.x, nextRobotCenter.y, nextRobotBottomLeft.d),
              {
                turnRadiusCells: DEFAULT_TURN_RADIUS_CELLS,
                commandPoseOverrides,
              }
            )
          );
          setTimeline(0);
        } else {
          setResponseCommands([]);
          setResponsePath([]);
          setSelectedViewStates([]);
          setTrajectory(
            buildTrajectoryFromCommands(
              [],
              createStartPose(nextRobotCenter.x, nextRobotCenter.y, nextRobotBottomLeft.d),
              {
                turnRadiusCells: DEFAULT_TURN_RADIUS_CELLS,
              }
            )
          );
          setTimeline(0);
          setErrorMessage(formatErrorMessage(result?.error));
        }
        setIsComputing(false);
      },
      "free-range"
    );
  };

  const onReset = () => {
    const nextRobot = {
      x: DEFAULT_ROBOT_BOTTOM_LEFT.x,
      y: DEFAULT_ROBOT_BOTTOM_LEFT.y,
      d: Direction.NORTH,
    };
    const center = robotBottomLeftToCenter(nextRobot.x, nextRobot.y);
    setRobotX(nextRobot.x);
    setRobotY(nextRobot.y);
    setRobotDir(Direction.NORTH);
    setStartRobot({ ...center, d: nextRobot.d });
    setErrorMessage("");
    resetTrajectoryState(center.x, center.y, nextRobot.d);
  };

  const onResetAll = () => {
    const nextRobot = {
      x: DEFAULT_ROBOT_BOTTOM_LEFT.x,
      y: DEFAULT_ROBOT_BOTTOM_LEFT.y,
      d: Direction.NORTH,
    };
    const center = robotBottomLeftToCenter(nextRobot.x, nextRobot.y);
    setRobotX(nextRobot.x);
    setRobotY(nextRobot.y);
    setRobotDir(Direction.NORTH);
    setStartRobot({ ...center, d: nextRobot.d });
    setObstacles([]);
    setObXInput(0);
    setObYInput(0);
    setDirectionInput(Direction.NORTH);
    setErrorMessage("");
    resetTrajectoryState(center.x, center.y, nextRobot.d);
  };

  const onPlayPause = () => {
    if (segmentCount === 0) {
      return;
    }

    if (isPlaying) {
      setIsPlaying(false);
      return;
    }

    if (timelinePos >= segmentCount) {
      setTimeline(0);
    }
    setIsPlaying(true);
  };

  const onPrev = () => {
    setIsPlaying(false);
    setTimeline(Math.max(currentFrame - 1, 0));
  };

  const onNext = () => {
    setIsPlaying(false);
    setTimeline(Math.min(currentFrame + 1, segmentCount));
  };

  useEffect(() => {
    timelineRef.current = timelinePos;
  }, [timelinePos]);

  useEffect(() => {
    if (!isPlaying) {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
        animationFrameRef.current = null;
      }
      lastTimestampRef.current = null;
      return;
    }

    if (segmentCount === 0) {
      setIsPlaying(false);
      return;
    }

    const animate = (timestamp) => {
      if (lastTimestampRef.current === null) {
        lastTimestampRef.current = timestamp;
      }

      const deltaSeconds = (timestamp - lastTimestampRef.current) / 1000;
      lastTimestampRef.current = timestamp;

      const nextTimeline = advanceTimeline(trajectory, timelineRef.current, deltaSeconds);
      timelineRef.current = nextTimeline;
      setTimelinePos(nextTimeline);

      if (nextTimeline >= segmentCount) {
        setIsPlaying(false);
        return;
      }

      animationFrameRef.current = requestAnimationFrame(animate);
    };

    animationFrameRef.current = requestAnimationFrame(animate);
    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
        animationFrameRef.current = null;
      }
      lastTimestampRef.current = null;
    };
  }, [isPlaying, segmentCount, trajectory]);

  const startCenter = toSvgPoint({ x: startRobot.x, y: startRobot.y });
  const robotCenter = toSvgPoint({ x: robotPose.x, y: robotPose.y });
  const robotAnchor = toSvgPoint(robotBottomLeftFromPose(robotPose.x, robotPose.y));
  const headingMarker = toSvgPoint({
    x: robotPose.x + HEADING_MARKER_FACTOR * Math.cos(robotPose.theta) * START_ZONE_RADIUS_CELLS,
    y: robotPose.y + HEADING_MARKER_FACTOR * Math.sin(robotPose.theta) * START_ZONE_RADIUS_CELLS,
  });

  return (
    <div className="flex flex-col items-center justify-center">
      <div className="flex flex-col items-center text-center bg-[#ddd6fe] rounded-xl shadow-xl mb-4">
        <h2 className="card-title text-black p-2 font-mono">
          ALGORITHM SIMULATOR (FREE-RANGE)
        </h2>
      </div>

      <div className="flex flex-col lg:flex-row items-start gap-8 w-full max-w-[1400px]">
        <div className="w-full lg:w-[min(66vw,700px)]">
          <div className="bg-white/80 rounded-xl shadow-xl p-3">
            <svg
              viewBox={`${WORLD_MIN} ${WORLD_MIN} ${WORLD_SIZE} ${WORLD_SIZE}`}
              className="w-full aspect-square rounded-lg bg-slate-100 border border-slate-300"
            >
              <rect
                x={WORLD_MIN}
                y={WORLD_MIN}
                width={WORLD_SIZE}
                height={WORLD_SIZE}
                fill="#f8fafc"
                stroke="#1e293b"
                strokeWidth={0.08}
              />

              {WORLD_TICKS.map((tick) => (
                <line
                  key={`vx-${tick}`}
                  x1={tick}
                  y1={WORLD_MIN}
                  x2={tick}
                  y2={WORLD_MAX}
                  stroke="#cbd5e1"
                  strokeWidth={0.03}
                />
              ))}
              {WORLD_TICKS.map((tick) => (
                <line
                  key={`hx-${tick}`}
                  x1={WORLD_MIN}
                  y1={tick}
                  x2={WORLD_MAX}
                  y2={tick}
                  stroke="#cbd5e1"
                  strokeWidth={0.03}
                />
              ))}

              <rect
                x={startCenter.x - START_ZONE_RADIUS_CELLS}
                y={startCenter.y - START_ZONE_RADIUS_CELLS}
                width={START_ZONE_RADIUS_CELLS * 2}
                height={START_ZONE_RADIUS_CELLS * 2}
                fill="#ede9fe"
                stroke="#8b5cf6"
                strokeWidth={0.05}
                opacity={0.7}
              />
              <text
                x={startCenter.x}
                y={startCenter.y + 0.2}
                textAnchor="middle"
                fontSize="0.45"
                fill="#5b21b6"
              >
                Start
              </text>

              {fullPathPolyline && (
                <polyline
                  points={fullPathPolyline}
                  fill="none"
                  stroke="#94a3b8"
                  strokeWidth={0.12}
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              )}

              {visitedPathPolyline && (
                <polyline
                  points={visitedPathPolyline}
                  fill="none"
                  stroke="#2563eb"
                  strokeWidth={0.16}
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              )}

              {obstacles.map((obstacle) => {
                const center = toSvgPoint(obstacleIndexToWorldCenter(obstacle));
                const face = obstacleFaceLine(obstacle);
                return (
                  <g key={`ob-${obstacle.id}`}>
                    <rect
                      x={center.x - 0.5}
                      y={center.y - 0.5}
                      width={1}
                      height={1}
                      fill="#1d4ed8"
                      stroke="#0f172a"
                      strokeWidth={0.04}
                    />
                    {face && (
                      <line
                        x1={face.x1}
                        y1={face.y1}
                        x2={face.x2}
                        y2={face.y2}
                        stroke="#ef4444"
                        strokeWidth={0.12}
                        strokeLinecap="round"
                      />
                    )}
                    <text
                      x={center.x}
                      y={center.y + 0.15}
                      textAnchor="middle"
                      fontSize="0.35"
                      fill="#f8fafc"
                    >
                      {obstacle.id}
                    </text>
                  </g>
                );
              })}

              <rect
                x={robotAnchor.x - ANCHOR_MARKER_HALF_WIDTH}
                y={robotAnchor.y - ANCHOR_MARKER_HALF_WIDTH}
                width={ANCHOR_MARKER_HALF_WIDTH * 2}
                height={ANCHOR_MARKER_HALF_WIDTH * 2}
                fill="#dc2626"
                stroke="#7f1d1d"
                strokeWidth={0.04}
              />

              <circle
                cx={robotCenter.x}
                cy={robotCenter.y}
                r={ROBOT_RADIUS_CELLS}
                fill={activeSegmentIndex >= 0 ? "#facc15" : "#fde68a"}
                stroke="#0f172a"
                strokeWidth={0.06}
              />
              <line
                x1={robotCenter.x}
                y1={robotCenter.y}
                x2={headingMarker.x}
                y2={headingMarker.y}
                stroke="#ef4444"
                strokeWidth={0.12}
                strokeLinecap="round"
              />
            </svg>

            <div className="flex justify-between text-sm text-slate-800 mt-2 font-mono">
              <span>World: 20 x 20</span>
              <span>Turn radius: {DEFAULT_TURN_RADIUS_CELLS.toFixed(1)} cells</span>
            </div>
            <div className="text-xs text-slate-600 mt-2 font-mono text-left">
              Origin input is bottom-left. Planner converts to center with +1,+1, so
              (0,0) becomes center (1,1) and occupies a 2-cell footprint (radius 1).
            </div>
          </div>
        </div>

        <div className="flex-1 w-full flex flex-col items-center text-center gap-6">
          <div className="flex flex-col items-center text-center bg-[#ddd6fe] rounded-xl shadow-xl w-full">
            <div className="card-body items-center text-center p-4 w-full">
              <h2 className="card-title text-black font-mono">Robot Position</h2>
              <div className="form-control w-full">
                <label className="input-group input-group-horizontal flex-wrap justify-center">
                  <span className="bg-primary p-2">X</span>
                  <input
                    onChange={onChangeRobotX}
                    type="number"
                    placeholder="0"
                    min="0"
                    max="17"
                    className="input input-bordered text-blue-900 w-20"
                  />
                  <span className="bg-primary p-2">Y</span>
                  <input
                    onChange={onChangeRobotY}
                    type="number"
                    placeholder="0"
                    min="0"
                    max="17"
                    className="input input-bordered text-blue-900 w-20"
                  />
                  <span className="bg-primary p-2">D</span>
                  <select
                    onChange={onRobotDirectionInputChange}
                    value={robotDir}
                    className="select text-blue-900 py-2 pl-2 pr-6"
                  >
                    <option value={Direction.NORTH}>Up</option>
                    <option value={Direction.SOUTH}>Down</option>
                    <option value={Direction.WEST}>Left</option>
                    <option value={Direction.EAST}>Right</option>
                  </select>
                  <button className="btn btn-success p-2" onClick={onClickRobot}>
                    Set
                  </button>
                </label>
              </div>
            </div>
          </div>

          <div className="flex flex-col items-center text-center bg-[#ddd6fe] p-4 rounded-xl shadow-xl w-full">
            <h2 className="card-title text-black pb-2 font-mono">Add Obstacles</h2>
            <div className="form-control w-full">
              <label className="input-group input-group-horizontal flex-wrap justify-center">
                <span className="bg-primary p-2">X</span>
                <input
                  onChange={onChangeX}
                  type="number"
                  placeholder="0"
                  min="0"
                  max="19"
                  className="input input-bordered text-blue-900 w-20"
                />
                <span className="bg-primary p-2">Y</span>
                <input
                  onChange={onChangeY}
                  type="number"
                  placeholder="0"
                  min="0"
                  max="19"
                  className="input input-bordered text-blue-900 w-20"
                />
                <span className="bg-primary p-2">D</span>
                <select
                  onChange={onDirectionInputChange}
                  value={directionInput}
                  className="select text-blue-900 py-2 pl-2 pr-6"
                >
                  <option value={Direction.NORTH}>Up</option>
                  <option value={Direction.SOUTH}>Down</option>
                  <option value={Direction.WEST}>Left</option>
                  <option value={Direction.EAST}>Right</option>
                  <option value={Direction.SKIP}>None</option>
                </select>
                <button className="btn btn-success p-2" onClick={onClickObstacle}>
                  Add
                </button>
              </label>
            </div>
          </div>

          <div className="grid grid-cols-4 gap-x-2 gap-y-3 items-center w-full">
            {obstacles.map((obstacle) => (
              <div
                key={`badge-${obstacle.id}`}
                className="badge flex flex-row text-black bg-sky-100 rounded-xl text-xs md:text-sm h-max border-cyan-500 cursor-pointer hover:bg-sky-200 transition-colors"
                onClick={() => onRemoveObstacle(obstacle)}
              >
                <div>
                  <div>X: {obstacle.x}</div>
                  <div>Y: {obstacle.y}</div>
                  <div>D: {DirectionToString[obstacle.d]}</div>
                </div>
                <div>
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    fill="none"
                    viewBox="0 0 24 24"
                    className="inline-block w-4 h-4 stroke-current"
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
            ))}
          </div>

          <div className="btn-group btn-group-horizontal py-2">
            <button className="btn btn-error" onClick={onResetAll}>
              Reset All
            </button>
            <button className="btn btn-warning" onClick={onReset}>
              Reset Robot
            </button>
            <button
              className={classNames("btn btn-success", isComputing ? "loading" : "")}
              onClick={compute}
            >
              {isComputing ? "Computing" : "Submit"}
            </button>
          </div>

          {(responseCommands.length > 0 || segmentCount > 0) && (
            <div className="flex flex-col items-center text-center bg-[#ddd6fe] p-4 rounded-xl shadow-xl my-2 gap-3 w-full">
              <div className="flex flex-row flex-wrap justify-center items-center gap-3">
                <button className="btn btn-success" onClick={onPlayPause}>
                  {isPlaying ? "Pause" : "Play"}
                </button>
                <button
                  className="btn btn-sm btn-outline"
                  onClick={onPrev}
                  disabled={currentFrame <= 0}
                >
                  Prev
                </button>
                <span className="text-black font-mono">
                  Step: {currentFrame} / {segmentCount}
                </span>
                <button
                  className="btn btn-sm btn-outline"
                  onClick={onNext}
                  disabled={currentFrame >= segmentCount}
                >
                  Next
                </button>
                <span className="text-black font-mono">Cmd: {currentCommand}</span>
              </div>
              <input
                type="range"
                min="0"
                max={segmentCount}
                value={currentFrame}
                onChange={(event) => {
                  setIsPlaying(false);
                  setTimeline(Number(event.target.value));
                }}
                className="range range-primary w-full"
              />
            </div>
          )}

          {responseCommands.length > 0 && (
            <div className="flex flex-col bg-[#ddd6fe] rounded-xl shadow-xl p-4 w-full gap-2 text-left">
              <div className="font-mono text-black">Commands</div>
              <div className="max-h-52 overflow-auto rounded border border-slate-200 bg-white/70 p-2 text-sm">
                {responseCommands.map((command, index) => (
                  <div
                    key={`cmd-${index}`}
                    className={classNames(
                      "px-2 py-1 rounded font-mono",
                      index === activeCommandIndex ? "bg-blue-200 text-blue-950" : ""
                    )}
                  >
                    {index + 1}. {command}
                  </div>
                ))}
              </div>
            </div>
          )}

          <div className="flex flex-col bg-[#ddd6fe] rounded-xl shadow-xl p-4 w-full gap-2 text-left text-sm">
            <div className="font-mono text-black">Debug</div>
            <div className="font-mono text-black">Motion segments: {segmentCount}</div>
            <div className="font-mono text-black">
              Path points from API: {responsePath.length}
            </div>
            <div className="font-mono text-black">
              Commands from API: {responseCommands.length}
            </div>
            <div className="font-mono text-black">
              Snap targets: {selectedViewStates.length}
            </div>
            <div className="font-mono text-black">
              Est. duration: {trajectory.totalDuration.toFixed(2)}s
            </div>
            {trajectory.warnings.length > 0 && (
              <div className="rounded bg-amber-50 border border-amber-300 p-2 text-amber-900 font-mono">
                {trajectory.warnings.map((warning, index) => (
                  <div key={`warn-${index}`}>{warning}</div>
                ))}
              </div>
            )}
            {errorMessage && (
              <div className="rounded bg-red-50 border border-red-300 p-2 text-red-900 font-mono">
                {errorMessage}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
