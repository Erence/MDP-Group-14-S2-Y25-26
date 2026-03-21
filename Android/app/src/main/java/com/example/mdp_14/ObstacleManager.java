package com.example.mdp_14;

import android.util.Log;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.util.List;

/**
 * Manages obstacle operations including:
 * - JSON formatting for transmission
 * - Obstacle validation
 * - Bulk operations
 */
public class ObstacleManager {
    private static final String TAG = "ObstacleManager";

    private final ArenaMapView arenaMapView;

    public ObstacleManager(ArenaMapView arenaMapView) {
        this.arenaMapView = arenaMapView;
    }

    // ============================================================
    // JSON BUILDING
    // ============================================================

    /**
     * Build JSON message containing all obstacles
     * Format: {"cat": "obstacles", "value": {"obstacles": [...], "mode": "0"}}
     */
    public JSONObject buildObstaclesJSON() throws JSONException {
        List<Obstacle> obstacles = arenaMapView.getObstacles();

        JSONObject message = new JSONObject();
        message.put("cat", "obstacles");

        JSONObject value = new JSONObject();
        JSONArray obstaclesArray = new JSONArray();

        for (Obstacle obs : obstacles) {
            obstaclesArray.put(formatObstacleJSON(obs));
        }

        value.put("obstacles", obstaclesArray);
        value.put("mode", "0");
        message.put("value", value);

        return message;
    }

    /**
     * Format single obstacle as JSON
     * Format: {"x": <x>, "y": <y>, "id": <id>, "d": <direction>}
     */
    private JSONObject formatObstacleJSON(Obstacle obs) throws JSONException {
        JSONObject json = new JSONObject();

        int direction = convertDirectionToNumeric(obs.getTargetFace());

        json.put("x", obs.getGridX());
        json.put("y", obs.getGridY());
        json.put("id", obs.getId());
        json.put("d", direction);

        return json;
    }

    /**
     * Convert direction enum to numeric value
     * NORTH: 0, EAST: 2, SOUTH: 4, WEST: 6
     */
    private int convertDirectionToNumeric(Obstacle.Direction direction) {
        switch (direction) {
            case EAST:  return 2;
            case SOUTH: return 4;
            case WEST:  return 6;
            case NORTH:
            default:    return 0;
        }
    }

    // ============================================================
    // OBSTACLE OPERATIONS
    // ============================================================

    /**
     * Find obstacle by ID
     */
    public Obstacle findObstacleById(int id) {
        for (Obstacle obs : arenaMapView.getObstacles()) {
            if (obs.getId() == id) {
                return obs;
            }
        }
        return null;
    }

    /**
     * Reset all obstacle recognitions
     */
    public void resetAllRecognitions() {
        List<Obstacle> obstacles = arenaMapView.getObstacles();

        for (Obstacle obstacle : obstacles) {
            obstacle.setRecognizedTargetId(null);
        }

        arenaMapView.invalidate();
        Log.d(TAG, "All obstacle recognitions cleared");
    }

    /**
     * Get obstacle count
     */
    public int getObstacleCount() {
        return arenaMapView.getObstacles().size();
    }

    /**
     * Check if obstacles exist
     */
    public boolean hasObstacles() {
        return !arenaMapView.getObstacles().isEmpty();
    }
}