package com.example.mdp_14;

import androidx.annotation.NonNull;

/**
 * Represents the robot on the arena map.
 * The robot is a fixed 2x2 size and has a facing direction.
 */
public class Robot {
    public static final int SIZE = 3; // Robot is always 3x3 grid units

    private int gridX;      // X position on grid (left edge)
    private int gridY;      // Y position on grid (bottom edge, since 0,0 is bottom-left)
    private Direction facing;

    public enum Direction {
        NORTH("N", 0),
        EAST("E", 2),
        SOUTH("S", 4),
        WEST("W", 6);

        private final String code;
        private final int numeric;

        Direction(String code, int numeric) {
            this.code = code;
            this.numeric = numeric;
        }

        public String getCode() {
            return code;
        }
        public int getNumeric() {
            return numeric;
        }

        public static Direction fromCode(String code) {
            for (Direction d : values()) {
                if (d.code.equalsIgnoreCase(code)) {
                    return d;
                }
            }
            return NORTH;
        }

        public static Direction fromNumeric(int numeric) {
            for (Direction d : values()) {
                if (d.numeric == numeric) {
                    return d;
                }
            }
            return NORTH;
        }
    }

    public Robot(int gridX, int gridY, Direction facing) {
        this.gridX = gridX;
        this.gridY = gridY;
        this.facing = facing;
    }

    public Robot() {
        // Default position: bottom-left corner, facing North
        // With (0,0) at bottom-left, the robot spawns at (0, 0)
        this.gridX = 0;
        this.gridY = 0;
        this.facing = Direction.NORTH;
    }

    public int getGridX() {
        return gridX;
    }

    public void setGridX(int gridX) {
        this.gridX = gridX;
    }

    public int getGridY() {
        return gridY;
    }

    public void setGridY(int gridY) {
        this.gridY = gridY;
    }

    public Direction getFacing() {
        return facing;
    }

    public void setFacing(Direction facing) {
        this.facing = facing;
    }

    /**
     * Check if a grid position is within the robot's bounds
     */
    public boolean containsPoint(int x, int y) {
        return x >= gridX && x < gridX + SIZE &&
                y >= gridY && y < gridY + SIZE;
    }

    /**
     * Rotate the robot 90 degrees clockwise
     */
    public void rotateClockwise() {
        int newNumeric = (facing.getNumeric() + 2) % 8;
        facing = Direction.fromNumeric(newNumeric);
    }

    /**
     * Rotate the robot 90 degrees counter-clockwise
     */
    public void rotateCounterClockwise() {
        int newNumeric = (facing.getNumeric() - 2 + 8) % 8;
        facing = Direction.fromNumeric(newNumeric);
    }

    @NonNull
    @Override
    public String toString() {
        return "Robot{pos=(" + gridX + "," + gridY + "), facing=" + facing + "}";
    }
}
