package com.example.mdp_14;

import android.util.Log;

import org.json.JSONException;
import org.json.JSONObject;

import java.util.ArrayList;
import java.util.List;

/**
 * Parses incoming JSON messages from the robot and delegates to appropriate handlers
 */
public class MessageParser {
    private static final String TAG = "MessageParser";

    private MessageCallback callback;

    public interface MessageCallback {
        void onStatusUpdate(String status);
        void onImageRecognition(String imageId, int obstacleId);
        void onRobotLocationUpdate(int x, int y, int direction);
        void onSimulationTrigger();
    }

    public MessageParser(MessageCallback callback) {
        this.callback = callback;
    }

    /**
     * Parse incoming message and delegate to appropriate handler
     */
    public void parseMessage(String message) {
        try {
            message = message.trim();

            // Find all JSON objects in the message
            List<String> jsonMessages = extractJsonObjects(message);

            for (String msg : jsonMessages) {
                parseSingleMessage(msg.trim());
            }
        } catch (Exception e) {
            Log.e(TAG, "Error parsing message: " + message, e);
        }
    }

    private List<String> extractJsonObjects(String message) {
        List<String> results = new ArrayList<>();
        int braceCount = 0;
        int startIndex = -1;

        for (int i = 0; i < message.length(); i++) {
            char c = message.charAt(i);

            if (c == '{') {
                if (braceCount == 0) {
                    startIndex = i; // Start of a JSON object
                }
                braceCount++;
            }else if (c == '}') {
                braceCount--;
                if (braceCount == 0 && startIndex != -1) {
                    // Complete JSON object found
                    String jsonObj = message.substring(startIndex, i + 1);
                    results.add(jsonObj);
                    startIndex = -1;
                }
            }
        }

        return results;
    }

    public void parseSingleMessage(String message) {
        try {

            if (message.trim().equalsIgnoreCase("simulation1")) {
                if (callback != null) {
                    callback.onSimulationTrigger();
                }
                return;
            }

            JSONObject json = new JSONObject(message);
            String category = json.getString("cat");

            switch (category) {
                case "status":
                    parseStatusMessage(message);
                    break;
                case "image-rec":
                    parseImageRecognitionMessage(message);
                    break;
                case "location":
                    parseLocationMessage(message);
                    break;
                case "info":
                    Log.d(TAG, "Info message received: " + message);
                    break;
                default:
                    Log.d(TAG, "Info message received: " + message);
                    break;
            }
        } catch (Exception e) {
            Log.e(TAG, "Error parsing message: " + message, e);
        }
    }

    /**
     * Parse STATUS message
     * Format: {"cat": "status", "value": <status>}
     */
    private void parseStatusMessage(String message) {
        try {
            JSONObject json = new JSONObject(message);
            String status = json.getString("value");

            if (callback != null) {
                callback.onStatusUpdate(status);
            }
        } catch (JSONException e) {
            Log.e(TAG, "Failed to parse status message: " + message, e);
        }
    }

    /**
     * Parse IMAGE-REC message
     * Format: {"cat": "image-rec", "value": {"image_id": <id>, "obstacle_id": <id>}}
     */
    private void parseImageRecognitionMessage(String message) {
        try {
            JSONObject json = new JSONObject(message);

            // Verify category
            if (!"image-rec".equals(json.getString("cat"))) {
                Log.w(TAG, "Unexpected message category: " + json.getString("cat"));
                return;
            }

            JSONObject value = json.getJSONObject("value");
            String imageId = value.getString("image_id");
            int obstacleId = value.getInt("obstacle_id");

            if (callback != null) {
                callback.onImageRecognition(imageId, obstacleId);
            }
        } catch (JSONException e) {
            Log.e(TAG, "Failed to parse image-rec message: " + message, e);
        }
    }

    /**
     * Parse LOCATION message
     * Format: {"cat": "location", "value": {"x": <x>, "y": <y>, "d": <direction>}}
     */
    private void parseLocationMessage(String message) {
        try {
            JSONObject json = new JSONObject(message);

            // Verify category
            if (!"location".equals(json.getString("cat"))) {
                Log.w(TAG, "Unexpected message category: " + json.getString("cat"));
                return;
            }

            JSONObject value = json.getJSONObject("value");
            int x = value.getInt("x");
            int y = value.getInt("y");
            int direction = value.getInt("d");

            if (callback != null) {
                callback.onRobotLocationUpdate(x, y, direction);
            }
        } catch (JSONException e) {
            Log.e(TAG, "Failed to parse location message: " + message, e);
        }
    }
}