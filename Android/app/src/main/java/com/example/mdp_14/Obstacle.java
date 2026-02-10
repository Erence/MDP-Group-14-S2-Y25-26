package com.example.mdp_14;

import java.io.Serializable;

/**
 * Represents an obstacle on the arena map.
 * Obstacles can have variable dimensions and a target image on one face.
 * The recognizedTargetId is set when the robot identifies the target via camera.
 */
public class Obstacle implements Serializable {
    private static int nextId = 1;

    private int id;
    private int gridX;              // X position on grid (left edge)
    private int gridY;              // Y position on grid (top edge)
    private int width;              // Width in grid units
    private int height;             // Height in grid units
    private Direction targetFace;   // Which face has the target image
    private String recognizedTargetId; // Set by robot when target is identified (C.9)

    public enum Direction {
        NORTH("North"),
        SOUTH("South"),
        EAST("East"),
        WEST("West");

        private final String displayName;

        Direction(String displayName) {
            this.displayName = displayName;
        }

        public String getDisplayName() {
            return displayName;
        }

        public static Direction fromDisplayName(String name) {
            for (Direction d : values()) {
                if (d.displayName.equalsIgnoreCase(name)) {
                    return d;
                }
            }
            return NORTH;
        }
    }

    public Obstacle(int gridX, int gridY) {
        this.id = nextId++;
        this.gridX = gridX;
        this.gridY = gridY;
        this.width = 1;
        this.height = 1;
        this.targetFace = Direction.NORTH;
        this.recognizedTargetId = null;
    }

    public Obstacle(int gridX, int gridY, int width, int height) {
        this.id = nextId++;
        this.gridX = gridX;
        this.gridY = gridY;
        this.width = width;
        this.height = height;
        this.targetFace = Direction.NORTH;
        this.recognizedTargetId = null;
    }

    // Getters and Setters
    public int getId() {
        return id;
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

    public int getWidth() {
        return width;
    }

    public void setWidth(int width) {
        this.width = Math.max(1, width);
    }

    public int getHeight() {
        return height;
    }

    public void setHeight(int height) {
        this.height = Math.max(1, height);
    }

    public Direction getTargetFace() {
        return targetFace;
    }

    public void setTargetFace(Direction targetFace) {
        this.targetFace = targetFace;
    }

    /**
     * Get the recognized target ID (set by robot via Bluetooth)
     */
    public String getRecognizedTargetId() {
        return recognizedTargetId;
    }

    /**
     * Set the recognized target ID (called when receiving TARGET message from robot)
     */
    public void setRecognizedTargetId(String recognizedTargetId) {
        this.recognizedTargetId = recognizedTargetId;
    }

    /**
     * Check if this obstacle has a recognized target ID
     */
    public boolean hasRecognizedTarget() {
        return recognizedTargetId != null && !recognizedTargetId.isEmpty();
    }

    /**
     * Check if a grid position is within this obstacle's bounds
     */
    public boolean containsPoint(int x, int y) {
        return x >= gridX && x < gridX + width &&
               y >= gridY && y < gridY + height;
    }

    /**
     * Check if this obstacle overlaps with another
     */
    public boolean overlaps(Obstacle other) {
        return !(gridX + width <= other.gridX ||
                 other.gridX + other.width <= gridX ||
                 gridY + height <= other.gridY ||
                 other.gridY + other.height <= gridY);
    }

    /**
     * Get the right edge X coordinate (exclusive)
     */
    public int getRightX() {
        return gridX + width;
    }

    /**
     * Get the bottom edge Y coordinate (exclusive)
     */
    public int getBottomY() {
        return gridY + height;
    }

    @Override
    public String toString() {
        return "Obstacle{" +
                "id=" + id +
                ", pos=(" + gridX + "," + gridY + ")" +
                ", size=" + width + "x" + height +
                ", face=" + targetFace +
                ", recognized=" + recognizedTargetId +
                '}';
    }

    /**
     * Reset the ID counter (useful for clearing all obstacles)
     */
    public static void resetIdCounter() {
        nextId = 1;
    }
}
