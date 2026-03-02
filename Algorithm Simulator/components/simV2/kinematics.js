const CM_PER_CELL = 10;
const EPSILON = 1e-9;

export const WORLD_SIZE = 20;
export const DEFAULT_TURN_RADIUS_CELLS = 1.8;
export const DEFAULT_STRAIGHT_SPEED_CELLS_PER_SEC = 3.2;
export const DEFAULT_ANGULAR_SPEED_RAD_PER_SEC = Math.PI / 2;
export const DEFAULT_LINE_SAMPLE_STEP_CELLS = 0.08;
export const DEFAULT_ARC_SAMPLE_STEP_RAD = (3 * Math.PI) / 180;
const MIN_SEGMENT_DURATION_SEC = 0.05;

const DIR_TO_THETA = {
  0: Math.PI / 2,
  2: 0,
  4: -Math.PI / 2,
  6: Math.PI,
};

const THETA_TO_DIR = [0, 2, 4, 6];

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function normalizeAngle(theta) {
  let normalized = theta;
  while (normalized <= -Math.PI) {
    normalized += Math.PI * 2;
  }
  while (normalized > Math.PI) {
    normalized -= Math.PI * 2;
  }
  return normalized;
}

function clonePose(pose) {
  return {
    x: pose.x,
    y: pose.y,
    theta: normalizeAngle(pose.theta),
  };
}

function appendUniquePoints(target, points) {
  if (!Array.isArray(points) || points.length === 0) {
    return;
  }

  for (const point of points) {
    if (target.length === 0) {
      target.push(point);
      continue;
    }

    const last = target[target.length - 1];
    if (
      Math.abs(last.x - point.x) > EPSILON ||
      Math.abs(last.y - point.y) > EPSILON
    ) {
      target.push(point);
    }
  }
}

function sampleLinePoints(startPose, endPose, lengthCells, stepCells) {
  if (lengthCells <= EPSILON) {
    return [{ x: startPose.x, y: startPose.y }];
  }

  const sampleCount = Math.max(2, Math.ceil(lengthCells / stepCells) + 1);
  const points = [];

  for (let i = 0; i < sampleCount; i += 1) {
    const t = i / (sampleCount - 1);
    points.push({
      x: startPose.x + (endPose.x - startPose.x) * t,
      y: startPose.y + (endPose.y - startPose.y) * t,
    });
  }

  return points;
}

function sampleArcPoints(
  centerX,
  centerY,
  radius,
  startAngle,
  endAngle,
  stepRad
) {
  const delta = endAngle - startAngle;
  if (Math.abs(delta) <= EPSILON) {
    return [
      {
        x: centerX + radius * Math.cos(startAngle),
        y: centerY + radius * Math.sin(startAngle),
      },
    ];
  }

  const sampleCount = Math.max(2, Math.ceil(Math.abs(delta) / stepRad) + 1);
  const points = [];

  for (let i = 0; i < sampleCount; i += 1) {
    const t = i / (sampleCount - 1);
    const angle = startAngle + delta * t;
    points.push({
      x: centerX + radius * Math.cos(angle),
      y: centerY + radius * Math.sin(angle),
    });
  }

  return points;
}

function parseCommandToken(rawCommand) {
  const normalized = String(rawCommand ?? "").trim().toUpperCase();
  if (!normalized) {
    return { type: "empty", raw: String(rawCommand ?? "") };
  }

  if (normalized.startsWith("SNAP")) {
    return { type: "snap", raw: normalized };
  }

  if (normalized === "FIN") {
    return { type: "fin", raw: normalized };
  }

  const match = /^(FW|BW|FR|FL|BR|BL)(\d+)$/.exec(normalized);
  if (!match) {
    return { type: "unknown", raw: normalized };
  }

  const opcode = match[1];
  const value = Number(match[2]);
  if (!Number.isFinite(value)) {
    return { type: "unknown", raw: normalized };
  }

  if (opcode === "FW" || opcode === "BW") {
    return {
      type: "motion",
      raw: normalized,
      opcode,
      motionType: "line",
      gearSign: opcode === "FW" ? 1 : -1,
      valueCm: value,
      distanceCells: value / CM_PER_CELL,
    };
  }

  return {
    type: "motion",
    raw: normalized,
    opcode,
    motionType: "arc",
    gearSign: opcode.startsWith("F") ? 1 : -1,
    steerSign: opcode.endsWith("L") ? 1 : -1,
    valueDeg: value,
    angleRad: (value * Math.PI) / 180,
  };
}

function makeLineSegment(
  startPose,
  parsed,
  options
) {
  const travelCells = parsed.distanceCells;
  if (travelCells <= EPSILON) {
    return null;
  }

  const signedDistance = parsed.gearSign * travelCells;
  const endPose = {
    x: startPose.x + signedDistance * Math.cos(startPose.theta),
    y: startPose.y + signedDistance * Math.sin(startPose.theta),
    theta: startPose.theta,
  };

  const duration = Math.max(
    MIN_SEGMENT_DURATION_SEC,
    travelCells / options.straightSpeedCellsPerSec
  );

  return {
    type: "line",
    gear: parsed.gearSign > 0 ? "forward" : "backward",
    startPose: clonePose(startPose),
    endPose: clonePose(endPose),
    length: travelCells,
    duration,
    samples: sampleLinePoints(
      startPose,
      endPose,
      travelCells,
      options.lineSampleStepCells
    ),
  };
}

function makeArcSegment(
  startPose,
  parsed,
  options
) {
  const turnAngle = parsed.angleRad;
  if (turnAngle <= EPSILON) {
    return null;
  }

  const deltaTheta = parsed.gearSign * parsed.steerSign * turnAngle;
  const radius = options.turnRadiusCells;

  const centerX = startPose.x - parsed.steerSign * radius * Math.sin(startPose.theta);
  const centerY = startPose.y + parsed.steerSign * radius * Math.cos(startPose.theta);
  const startAngle = Math.atan2(startPose.y - centerY, startPose.x - centerX);
  const endAngle = startAngle + deltaTheta;

  const endPose = {
    x: centerX + radius * Math.cos(endAngle),
    y: centerY + radius * Math.sin(endAngle),
    theta: normalizeAngle(startPose.theta + deltaTheta),
  };

  const duration = Math.max(
    MIN_SEGMENT_DURATION_SEC,
    turnAngle / options.angularSpeedRadPerSec
  );

  return {
    type: "arc",
    gear: parsed.gearSign > 0 ? "forward" : "backward",
    steer: parsed.steerSign > 0 ? "left" : "right",
    startPose: clonePose(startPose),
    endPose: clonePose(endPose),
    center: { x: centerX, y: centerY },
    radius,
    startAngle,
    endAngle,
    deltaTheta,
    length: radius * turnAngle,
    duration,
    samples: sampleArcPoints(
      centerX,
      centerY,
      radius,
      startAngle,
      endAngle,
      options.arcSampleStepRad
    ),
  };
}

function sanitizeOptions(options = {}) {
  return {
    turnRadiusCells:
      options.turnRadiusCells ?? DEFAULT_TURN_RADIUS_CELLS,
    straightSpeedCellsPerSec:
      options.straightSpeedCellsPerSec ?? DEFAULT_STRAIGHT_SPEED_CELLS_PER_SEC,
    angularSpeedRadPerSec:
      options.angularSpeedRadPerSec ?? DEFAULT_ANGULAR_SPEED_RAD_PER_SEC,
    lineSampleStepCells:
      options.lineSampleStepCells ?? DEFAULT_LINE_SAMPLE_STEP_CELLS,
    arcSampleStepRad:
      options.arcSampleStepRad ?? DEFAULT_ARC_SAMPLE_STEP_RAD,
    commandPoseOverrides: options.commandPoseOverrides ?? {},
  };
}

export function directionToTheta(direction) {
  return DIR_TO_THETA[direction] ?? Math.PI / 2;
}

export function thetaToDirection(theta) {
  const normalized = normalizeAngle(theta);
  const candidates = [
    { dir: 0, theta: Math.PI / 2 },
    { dir: 2, theta: 0 },
    { dir: 4, theta: -Math.PI / 2 },
    { dir: 6, theta: Math.PI },
  ];

  let bestDir = 0;
  let bestDelta = Number.POSITIVE_INFINITY;
  for (const candidate of candidates) {
    const delta = Math.abs(normalizeAngle(normalized - candidate.theta));
    if (delta < bestDelta) {
      bestDelta = delta;
      bestDir = candidate.dir;
    }
  }

  return bestDir;
}

export function createStartPose(x, y, direction) {
  return {
    x: Number(x),
    y: Number(y),
    theta: directionToTheta(Number(direction)),
  };
}

export function buildTrajectoryFromCommands(commands, startPose, options) {
  const normalizedOptions = sanitizeOptions(options);
  const safeCommands = Array.isArray(commands) ? commands : [];
  const safeStartPose = {
    x: Number.isFinite(startPose?.x) ? Number(startPose.x) : 1,
    y: Number.isFinite(startPose?.y) ? Number(startPose.y) : 1,
    theta: Number.isFinite(startPose?.theta)
      ? normalizeAngle(Number(startPose.theta))
      : Math.PI / 2,
  };

  const segments = [];
  const commandItems = [];
  const warnings = [];
  let currentPose = clonePose(safeStartPose);

  safeCommands.forEach((rawCommand, commandIndex) => {
    const parsed = parseCommandToken(rawCommand);
    const item = {
      index: commandIndex,
      raw: parsed.raw,
      type: parsed.type,
      motionSegmentIndex: null,
    };

    if (parsed.type === "motion") {
      const segment =
        parsed.motionType === "line"
          ? makeLineSegment(currentPose, parsed, normalizedOptions)
          : makeArcSegment(currentPose, parsed, normalizedOptions);

      if (segment) {
        segment.index = segments.length;
        segment.command = parsed.raw;
        segment.commandIndex = commandIndex;

        const overridePose = normalizedOptions.commandPoseOverrides?.[commandIndex];
        if (overridePose && typeof overridePose === "object") {
          if (Number.isFinite(overridePose.x)) {
            segment.endPose.x = Number(overridePose.x);
          }
          if (Number.isFinite(overridePose.y)) {
            segment.endPose.y = Number(overridePose.y);
          }
          if (Number.isFinite(overridePose.theta)) {
            segment.endPose.theta = normalizeAngle(Number(overridePose.theta));
          }
        }

        segments.push(segment);
        item.motionSegmentIndex = segment.index;
        currentPose = clonePose(segment.endPose);
      } else {
        warnings.push(`Skipped zero-length motion command: ${parsed.raw}`);
      }
    } else if (parsed.type === "unknown") {
      warnings.push(`Skipped unknown command token: ${parsed.raw}`);
    } else if (parsed.type === "empty") {
      warnings.push("Skipped empty command token");
    }

    commandItems.push(item);
  });

  const totalDuration = segments.reduce((sum, segment) => sum + segment.duration, 0);
  const totalLength = segments.reduce((sum, segment) => sum + segment.length, 0);

  return {
    startPose: safeStartPose,
    endPose: clonePose(currentPose),
    segments,
    commandItems,
    warnings,
    totalDuration,
    totalLength,
    options: normalizedOptions,
  };
}

export function frameAtTimeline(trajectory, timelinePos) {
  const segmentCount = trajectory?.segments?.length ?? 0;
  return Math.floor(clamp(timelinePos + EPSILON, 0, segmentCount));
}

export function activeSegmentAtTimeline(trajectory, timelinePos) {
  const segmentCount = trajectory?.segments?.length ?? 0;
  if (segmentCount === 0 || timelinePos <= 0) {
    return -1;
  }

  if (timelinePos >= segmentCount) {
    return segmentCount - 1;
  }

  const segmentIndex = Math.floor(timelinePos);
  const localT = timelinePos - segmentIndex;
  if (localT <= EPSILON) {
    return Math.max(segmentIndex - 1, 0);
  }

  return segmentIndex;
}

export function interpolatePoseOnSegment(segment, tValue) {
  const t = clamp(tValue, 0, 1);
  if (segment.type === "line") {
    return {
      x: segment.startPose.x + (segment.endPose.x - segment.startPose.x) * t,
      y: segment.startPose.y + (segment.endPose.y - segment.startPose.y) * t,
      theta: segment.startPose.theta,
    };
  }

  const angle = segment.startAngle + (segment.endAngle - segment.startAngle) * t;
  return {
    x: segment.center.x + segment.radius * Math.cos(angle),
    y: segment.center.y + segment.radius * Math.sin(angle),
    theta: normalizeAngle(segment.startPose.theta + segment.deltaTheta * t),
  };
}

export function poseAtTimeline(trajectory, timelinePos) {
  const segments = trajectory?.segments ?? [];
  const startPose = trajectory?.startPose ?? {
    x: 1,
    y: 1,
    theta: Math.PI / 2,
  };
  if (segments.length === 0 || timelinePos <= 0) {
    return startPose;
  }

  const capped = clamp(timelinePos, 0, segments.length);
  if (capped >= segments.length) {
    return segments[segments.length - 1].endPose;
  }

  const segmentIndex = Math.floor(capped);
  const localT = capped - segmentIndex;
  if (localT <= EPSILON) {
    if (segmentIndex === 0) {
      return startPose;
    }
    return segments[segmentIndex - 1].endPose;
  }

  return interpolatePoseOnSegment(segments[segmentIndex], localT);
}

export function advanceTimeline(trajectory, timelinePos, deltaSeconds) {
  const segments = trajectory?.segments ?? [];
  const segmentCount = segments.length;
  if (segmentCount === 0) {
    return 0;
  }

  let nextTimeline = clamp(timelinePos, 0, segmentCount);
  let remaining = Math.max(deltaSeconds, 0);

  while (remaining > EPSILON && nextTimeline < segmentCount) {
    const segmentIndex = Math.floor(nextTimeline);
    if (segmentIndex >= segmentCount) {
      return segmentCount;
    }

    const segment = segments[segmentIndex];
    const localT = nextTimeline - segmentIndex;
    const remainingSegmentTime = segment.duration * (1 - localT);
    if (remaining >= remainingSegmentTime) {
      remaining -= remainingSegmentTime;
      nextTimeline = segmentIndex + 1;
    } else {
      nextTimeline += remaining / segment.duration;
      remaining = 0;
    }
  }

  return clamp(nextTimeline, 0, segmentCount);
}

export function allSamplePoints(trajectory) {
  const points = [];
  const segments = trajectory?.segments ?? [];
  segments.forEach((segment) => {
    appendUniquePoints(points, segment.samples);
  });
  return points;
}

export function visitedSamplePointsAtTimeline(trajectory, timelinePos) {
  const segments = trajectory?.segments ?? [];
  const result = [];
  if (segments.length === 0 || timelinePos <= 0) {
    return result;
  }

  const capped = clamp(timelinePos, 0, segments.length);
  const fullSegments = Math.floor(capped);
  for (let i = 0; i < fullSegments; i += 1) {
    appendUniquePoints(result, segments[i].samples);
  }

  const localT = capped - fullSegments;
  if (localT > EPSILON && fullSegments < segments.length) {
    const segmentSamples = segments[fullSegments].samples ?? [];
    if (segmentSamples.length > 0) {
      const sampleIndex = Math.max(
        1,
        Math.floor(localT * (segmentSamples.length - 1))
      );
      appendUniquePoints(result, segmentSamples.slice(0, sampleIndex + 1));
    }
  }

  return result;
}

export function toSvgPoint(point) {
  return {
    x: point.x,
    y: WORLD_SIZE - 1 - point.y,
  };
}

export function toSvgPolyline(points) {
  if (!Array.isArray(points) || points.length === 0) {
    return "";
  }

  return points
    .map((point) => {
      const svgPoint = toSvgPoint(point);
      return `${svgPoint.x},${svgPoint.y}`;
    })
    .join(" ");
}

export const DIRECTIONS = THETA_TO_DIR;
