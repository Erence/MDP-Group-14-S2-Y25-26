package com.example.mdp_14;

import android.Manifest;
import android.bluetooth.BluetoothAdapter;
import android.bluetooth.BluetoothDevice;
import android.bluetooth.BluetoothSocket;
import android.content.Context;
import android.content.pm.PackageManager;
import android.content.res.ColorStateList;
import android.graphics.Color;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.os.Message;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageButton;
import android.widget.Spinner;
import android.widget.Switch;
import android.widget.TextView;
import android.widget.Toast;
import android.widget.ToggleButton;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Locale;
import java.util.Objects;
import java.util.Set;
import java.util.UUID;

public class MainActivity extends AppCompatActivity implements SensorEventListener, ArenaMapView.OnObstacleActionListener {
    private static final String TAG = "MainActivity";
    private static final int REQUEST_BLUETOOTH_PERMISSIONS = 1;
    private static final UUID MY_UUID = UUID.fromString("00001101-0000-1000-8000-00805F9B34FB");

    // Menu items for ActionBar
    private MenuItem deviceNameMenuItem;
    private String connectedDeviceName = null;
    private Button connectButton;

    // UI Elements - Status displays
    private TextView robotStatusText;
    private TextView positionText;
    private TextView directionText;
    private TextView receivedText;
    private EditText messageInput;
    private ImageButton sendButton;
    private Button clearMessagesButton;

    // UI Elements - D-Pad controls (C.3)
    private Button upButton;
    private Button downButton;
    private Button leftButton;
    private Button rightButton;
    private Switch tiltControlSwitch;

    // UI Elements - Arena Map (C.5, C.6, C.7)
    private ArenaMapView arenaMapView;
    private Button addObstacleButton;
    private Button editObstacleButton;
    private Button deleteObstacleButton;
    private Button clearAllButton;
    private Button spawnRobotButton;
    private Button sendObstaclesButton;
    private Button startRobotButton;
    private ToggleButton lockToggle;

    // Bluetooth
    private BluetoothAdapter bluetoothAdapter;
    private BluetoothService bluetoothService;
    private boolean isConnected = false;

    // Tilt control variables (C.3)
    private SensorManager sensorManager;
    private Sensor accelerometer;
    private boolean isTiltControlEnabled = false;
    private long lastTiltCommandTime = 0;
    private static final long TILT_COMMAND_INTERVAL = 500;
    private static final float TILT_THRESHOLD = 3.0f;

    /*
     * ============================================================
     * BLUETOOTH PROTOCOL FOR OBSTACLES (C.6 & C.7)
     * ============================================================
     *
     * Format for sending all obstacles:
     *   OBSTACLES,<count>
     *   OBS,<id>,<x>,<y>,<width>,<height>,<target_face>
     *   OBS,<id>,<x>,<y>,<width>,<height>,<target_face>
     *   ...
     *
     * Where:
     *   <id>          = Obstacle number (1, 2, 3, ...)
     *   <x>, <y>      = Grid coordinates (0-19)
     *   <width>       = Width in grid units
     *   <height>      = Height in grid units
     *   <target_face> = N, S, E, W (which face has the target image)
     *
     * Example:
     *   OBSTACLES,3
     *   OBS,1,5,10,1,1,N
     *   OBS,2,8,3,2,1,E
     *   OBS,3,15,15,1,2,S
     *
     * ============================================================
     */

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // Initialize UI elements
        initializeViews();

        // Initialize sensor manager for tilt control (C.3)
        sensorManager = (SensorManager) getSystemService(Context.SENSOR_SERVICE);
        accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);

        // Get Bluetooth adapter
        bluetoothAdapter = BluetoothAdapter.getDefaultAdapter();

        if (bluetoothAdapter == null) {
            Toast.makeText(this, "Bluetooth is not available on this device", Toast.LENGTH_LONG).show();
            finish();
            return;
        }

        // Initialize Bluetooth service
        bluetoothService = new BluetoothService(messageHandler, bluetoothAdapter);

        // Set up button listeners
        setupDPadControls();
        setupArenaMapUI();

        // Check permissions on startup
        checkPermissions();

        // Auto-start listening for incoming connections
        startListeningOnStartup();
    }

    private void initializeViews() {
        // Status displays (C.4)
        robotStatusText = findViewById(R.id.robotStatusTxt);
        positionText = findViewById(R.id.positionTxt);
        directionText = findViewById(R.id.directionTxt);
        receivedText = findViewById(R.id.receivedText);
        messageInput = findViewById(R.id.messageInput);
        sendButton = findViewById(R.id.sendBtn);
        clearMessagesButton = findViewById(R.id.clearMessagesBtn);

        // D-Pad controls (C.3)
        upButton = findViewById(R.id.upBtn);
        downButton = findViewById(R.id.downBtn);
        leftButton = findViewById(R.id.leftBtn);
        rightButton = findViewById(R.id.rightBtn);
        tiltControlSwitch = findViewById(R.id.tiltControlSwitch);

        // Arena map views (C.5, C.6, C.7)
        arenaMapView = findViewById(R.id.arenaMapView);
        addObstacleButton = findViewById(R.id.addObstacleButton);
        editObstacleButton = findViewById(R.id.editObstacleButton);
        deleteObstacleButton = findViewById(R.id.deleteObstacleButton);
        clearAllButton = findViewById(R.id.clearAllButton);
        spawnRobotButton = findViewById(R.id.spawnRobotButton);
        sendObstaclesButton = findViewById(R.id.sendObstaclesButton);
        startRobotButton = findViewById(R.id.startRobotButton);
        lockToggle = findViewById(R.id.lockToggle);
    }

    /**
     * Setup D-Pad controls and message sending (C.3)
     */
    private void setupDPadControls() {
        // Direction button listeners
        upButton.setOnClickListener(v -> sendCommand("move:up"));
        downButton.setOnClickListener(v -> sendCommand("move:down"));
        leftButton.setOnClickListener(v -> sendCommand("move:left"));
        rightButton.setOnClickListener(v -> sendCommand("move:right"));

        // Custom message send button
        sendButton.setOnClickListener(v -> {
            String message = messageInput.getText().toString().trim();
            if (!message.isEmpty()) {
                sendCommand(message);
                messageInput.setText("");
            } else {
                Toast.makeText(this, "Please enter a message", Toast.LENGTH_SHORT).show();
            }
        });

        // Tilt control switch listener
        tiltControlSwitch.setOnCheckedChangeListener((buttonView, isChecked) -> {
            if (isChecked && isConnected) {
                enableTiltControl();
            } else {
                disableTiltControl();
            }
        });

        clearMessagesButton.setOnClickListener(v -> clearMessages());
    }

    private void setupArenaMapUI() {
        arenaMapView.setOnObstacleActionListener(this);

        // Lock toggle - controls whether elements can be dragged
        lockToggle.setOnCheckedChangeListener((buttonView, isChecked) -> {
            arenaMapView.setDragLocked(isChecked);
            if (isChecked) {
                Toast.makeText(this, "Map locked - dragging disabled", Toast.LENGTH_SHORT).show();
            } else {
                Toast.makeText(this, "Map unlocked - dragging enabled", Toast.LENGTH_SHORT).show();
            }
        });

        // Spawn robot button
        spawnRobotButton.setOnClickListener(v -> {
            if (arenaMapView.hasRobot()) {
                new AlertDialog.Builder(this)
                        .setTitle("Robot")
                        .setMessage("Robot already exists. What would you like to do?")
                        .setPositiveButton("Reset Position", (dialog, which) -> {
                            arenaMapView.spawnRobot();
                            Toast.makeText(this, "Robot reset to start position", Toast.LENGTH_SHORT).show();
                        })
                        .setNegativeButton("Remove", (dialog, which) -> {
                            arenaMapView.removeRobot();
                            Toast.makeText(this, "Robot removed", Toast.LENGTH_SHORT).show();
                        })
                        .setNeutralButton("Cancel", null)
                        .show();
            } else {
                arenaMapView.spawnRobot();
                Toast.makeText(this, "Robot spawned at bottom-left. Drag to position.", Toast.LENGTH_SHORT).show();
            }
        });

        // Send obstacles button (C.6 & C.7)
        sendObstaclesButton.setOnClickListener(v -> {
            try {
                sendAllObstaclesToRobot();
            } catch (JSONException e) {
                throw new RuntimeException(e);
            }
        });

        startRobotButton.setOnClickListener(v -> {
            sendCommand("{\"cat\": \"control\", \"value\": \"start\"}");
        });

        addObstacleButton.setOnClickListener(v -> showAddObstacleDialog());

        editObstacleButton.setOnClickListener(v -> {
            Obstacle selected = arenaMapView.getSelectedObstacle();
            if (selected != null) {
                showEditObstacleDialog(selected);
            } else {
                Toast.makeText(this, "Please select an obstacle first", Toast.LENGTH_SHORT).show();
            }
        });

        deleteObstacleButton.setOnClickListener(v -> {
            Obstacle selected = arenaMapView.getSelectedObstacle();
            if (selected != null) {
                arenaMapView.removeObstacle(selected);
                Toast.makeText(this, "Obstacle deleted", Toast.LENGTH_SHORT).show();
            } else {
                Toast.makeText(this, "Please select an obstacle first", Toast.LENGTH_SHORT).show();
            }
        });

        clearAllButton.setOnClickListener(v -> {
            new AlertDialog.Builder(this)
                    .setTitle("Clear All")
                    .setMessage("Remove all obstacles and robot?")
                    .setPositiveButton("Yes", (dialog, which) -> {
                        arenaMapView.clearObstacles();
                        arenaMapView.removeRobot();
                        Toast.makeText(this, "All cleared", Toast.LENGTH_SHORT).show();
                    })
                    .setNegativeButton("No", null)
                    .show();
        });
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        getMenuInflater().inflate(R.menu.main_menu, menu);
        deviceNameMenuItem = menu.findItem(R.id.deviceNameTxt);
        MenuItem item = menu.findItem(R.id.connectBtn);
        connectButton = Objects.requireNonNull(item.getActionView()).findViewById(R.id.connectBtn);
        connectButton.setOnClickListener(v -> {
            if (!isConnected) {
                checkPermissionsAndConnect();
            } else {
                disconnect();
            }
        });
        updateActionBarMenuItem();
        return true;
    }

    private void updateActionBarMenuItem() {
        if (deviceNameMenuItem != null) {
            if (isConnected && connectedDeviceName != null) {
                deviceNameMenuItem.setTitle(connectedDeviceName);
                deviceNameMenuItem.setVisible(true);
                connectButton.setText("Disconnect");
                connectButton.setBackgroundTintList(ColorStateList.valueOf(Color.parseColor("#F44336"))); //Red
            } else {
                deviceNameMenuItem.setVisible(false);
                connectButton.setText("Connect");
                connectButton.setBackgroundTintList(ColorStateList.valueOf(Color.parseColor("#4CAF50"))); //Green
            }
        }
    }

    // ============================================================
    // C.6 & C.7: OBSTACLE BLUETOOTH TRANSMISSION
    // ============================================================

    private void sendAllObstaclesToRobot() throws JSONException {
        List<Obstacle> obstacles = arenaMapView.getObstacles();

        if (!isConnected) {
            Toast.makeText(this, "Not connected to robot", Toast.LENGTH_SHORT).show();
            return;
        }

        if (obstacles.isEmpty()) {
            Toast.makeText(this, "No obstacles to send", Toast.LENGTH_SHORT).show();
            return;
        }

        // Build JSON object with all obstacles
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

        sendCommand(message.toString());

        Toast.makeText(this, "Sent " + obstacles.size() + " obstacle(s) to robot", Toast.LENGTH_SHORT).show();
        Log.d(TAG, "Sent " + obstacles.size() + " obstacles to robot");
    }

    public JSONObject formatObstacleJSON(Obstacle obs) throws JSONException {
        JSONObject json = new JSONObject();
        int direction;

        switch(obs.getTargetFace().toString()){
            case"EAST":
                direction = 2;
                break;
            case "SOUTH":
                direction = 4;
                break;
            case"WEST":
                direction = 6;
                break;
            default:
                direction = 0;
        }

        json.put("x", obs.getGridX());
        json.put("y", obs.getGridY());
        json.put("id", obs.getId());
        json.put("d", direction);
        return json;
    }

    // ============================================================
    // OBSTACLE DIALOGS
    // ============================================================

    private void showAddObstacleDialog() {
        View dialogView = LayoutInflater.from(this).inflate(R.layout.dialog_edit_obstacle, null);

        TextView titleText = dialogView.findViewById(R.id.obstacleIdText);
        titleText.setText("Add New Obstacle");

        EditText widthInput = dialogView.findViewById(R.id.widthInput);
        EditText heightInput = dialogView.findViewById(R.id.heightInput);
        Spinner faceSpinner = dialogView.findViewById(R.id.faceSpinner);

        String[] directions = {"North", "South", "East", "West"};
        ArrayAdapter<String> adapter = new ArrayAdapter<>(this,
                android.R.layout.simple_spinner_item, directions);
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        faceSpinner.setAdapter(adapter);

        widthInput.setText("1");
        heightInput.setText("1");

        new AlertDialog.Builder(this)
                .setTitle("Add Obstacle")
                .setView(dialogView)
                .setPositiveButton("Add", (dialog, which) -> {
                    try {
                        int width = Integer.parseInt(widthInput.getText().toString());
                        int height = Integer.parseInt(heightInput.getText().toString());
                        String selectedFace = (String) faceSpinner.getSelectedItem();

                        width = Math.max(1, Math.min(width, arenaMapView.getGridSize()));
                        height = Math.max(1, Math.min(height, arenaMapView.getGridSize()));

                        int gridX = (arenaMapView.getGridSize() - width) / 2;
                        int gridY = (arenaMapView.getGridSize() - height) / 2;

                        Obstacle obstacle = new Obstacle(gridX, gridY, width, height);
                        obstacle.setTargetFace(Obstacle.Direction.fromDisplayName(selectedFace));

                        arenaMapView.addObstacle(obstacle);
                        arenaMapView.setSelectedObstacle(obstacle);

                        Toast.makeText(this, "Obstacle added. Drag to position.", Toast.LENGTH_SHORT).show();
                    } catch (NumberFormatException e) {
                        Toast.makeText(this, "Invalid dimensions", Toast.LENGTH_SHORT).show();
                    }
                })
                .setNegativeButton("Cancel", null)
                .show();
    }

    private void showEditObstacleDialog(Obstacle obstacle) {
        View dialogView = LayoutInflater.from(this).inflate(R.layout.dialog_edit_obstacle, null);

        TextView titleText = dialogView.findViewById(R.id.obstacleIdText);
        titleText.setText("Obstacle #" + obstacle.getId());

        EditText widthInput = dialogView.findViewById(R.id.widthInput);
        EditText heightInput = dialogView.findViewById(R.id.heightInput);
        Spinner faceSpinner = dialogView.findViewById(R.id.faceSpinner);

        String[] directions = {"North", "South", "East", "West"};
        ArrayAdapter<String> adapter = new ArrayAdapter<>(this,
                android.R.layout.simple_spinner_item, directions);
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        faceSpinner.setAdapter(adapter);

        widthInput.setText(String.valueOf(obstacle.getWidth()));
        heightInput.setText(String.valueOf(obstacle.getHeight()));

        for (int i = 0; i < directions.length; i++) {
            if (directions[i].equals(obstacle.getTargetFace().getDisplayName())) {
                faceSpinner.setSelection(i);
                break;
            }
        }

        new AlertDialog.Builder(this)
                .setTitle("Edit Obstacle")
                .setView(dialogView)
                .setPositiveButton("Save", (dialog, which) -> {
                    try {
                        int width = Integer.parseInt(widthInput.getText().toString());
                        int height = Integer.parseInt(heightInput.getText().toString());
                        String selectedFace = (String) faceSpinner.getSelectedItem();

                        width = Math.max(1, Math.min(width, arenaMapView.getGridSize()));
                        height = Math.max(1, Math.min(height, arenaMapView.getGridSize()));

                        obstacle.setWidth(width);
                        obstacle.setHeight(height);
                        obstacle.setTargetFace(Obstacle.Direction.fromDisplayName(selectedFace));

                        if (obstacle.getGridX() + width > arenaMapView.getGridSize()) {
                            obstacle.setGridX(arenaMapView.getGridSize() - width);
                        }
                        if (obstacle.getGridY() + height > arenaMapView.getGridSize()) {
                            obstacle.setGridY(arenaMapView.getGridSize() - height);
                        }

                        arenaMapView.updateObstacle(obstacle);
                        Toast.makeText(this, "Obstacle updated", Toast.LENGTH_SHORT).show();
                    } catch (NumberFormatException e) {
                        Toast.makeText(this, "Invalid dimensions", Toast.LENGTH_SHORT).show();
                    }
                })
                .setNegativeButton("Cancel", null)
                .show();
    }

    // ============================================================
    // ArenaMapView.OnObstacleActionListener implementation
    // ============================================================

    @Override
    public void onObstacleLongPress(Obstacle obstacle) {
        showEditObstacleDialog(obstacle);
    }

    @Override
    public void onObstacleSelected(Obstacle obstacle) {
        Log.d(TAG, "Selected obstacle: " + obstacle);
    }

    @Override
    public void onObstaclePositionChanged(Obstacle obstacle) {
        Log.d(TAG, "Obstacle moved: " + obstacle);
    }

    @Override
    public void onObstacleRemovedByDrag(Obstacle obstacle) {
        Log.d(TAG, "Obstacle removed by drag: " + obstacle);
        Toast.makeText(this, "Obstacle #" + obstacle.getId() + " removed", Toast.LENGTH_SHORT).show();
    }

    @Override
    public void onRobotPositionChanged(Robot robot) {
        Log.d(TAG, "Robot moved: " + robot);
        // Update position and direction displays
        if (robot != null) {
            positionText.setText(robot.getGridX() + "," + robot.getGridY());
            directionText.setText(robot.getFacing().name());
        }
    }

    @Override
    public void onEmptyCellTap(int gridX, int gridY) {
        Log.d(TAG, "Empty cell tapped: (" + gridX + ", " + gridY + ")");
    }

    // ============================================================
    // BLUETOOTH CONNECTION MANAGEMENT
    // ============================================================

    private void startListeningOnStartup() {
        new Handler(Looper.getMainLooper()).postDelayed(() -> {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
                if (ContextCompat.checkSelfPermission(this, Manifest.permission.BLUETOOTH_CONNECT)
                        != PackageManager.PERMISSION_GRANTED) {
                    Log.d(TAG, "Bluetooth permission not granted yet");
                    return;
                }
            }

            if (!isConnected) {
                bluetoothService.startServer();
                Log.d(TAG, "Started listening for incoming connections");
            }
        }, 500);
    }

    private final Handler messageHandler = new Handler(Looper.getMainLooper()) {
        @Override
        public void handleMessage(@NonNull Message msg) {
            switch (msg.what) {
                case BluetoothService.MESSAGE_READ:
                    String receivedMessage = (String) msg.obj;
                    handleIncomingMessage(receivedMessage);
                    break;

                case BluetoothService.MESSAGE_WRITE:
                    String sentMessage = (String) msg.obj;
                    logMessage("Sent: " + sentMessage, "#1976D2");
                    break;

                case BluetoothService.MESSAGE_DISCONNECTED:
                    handleDisconnection();
                    break;

                case BluetoothService.MESSAGE_CONNECTED:
                    handleIncomingConnection();
                    break;
            }
        }
    };

    // ============================================================
    // MESSAGE PARSING (C.4, C.9, C.10)
    // ============================================================

    private void handleIncomingMessage(String message) {
        logMessage("Received: " + message, "#388E3C");

        // Parse different message types
        if (message.contains("status")) {
            handleStatusUpdate(message);
        } else if (message.contains("image-rec")) {
            handleTargetMessage(message);
        } else if (message.contains("location")) {
            handleRobotMessage(message);
        }
    }

    /**
     * Handle STATUS message (C.4)
     */
    private void handleStatusUpdate(String message) {
        try {
            JSONObject json = new JSONObject(message);
            String statusMsg = json.getString("status").toUpperCase();
            robotStatusText.setText(statusMsg);
        } catch (JSONException e) {
            Log.d(TAG, "Not a JSON status message: " + message);
        }
    }

    /**
     * Handle TARGET message (C.9)
     * Format: {"cat": "image-rec", "value": {"image_id": <Target ID>, "obstacle_id":  <Obstacle Number>}}
     */
    private void handleTargetMessage(String message) {
        try {
            JSONObject json = new JSONObject(message);

            JSONObject value = json.getJSONObject("value");
            String targetId = value.getString("image_id");
            int obstacleNumber = value.getInt("obstacle_id");

            Obstacle obstacle = findObstacleById(obstacleNumber);
            if (obstacle != null) {
                obstacle.setRecognizedTargetId(targetId);
                arenaMapView.updateObstacle(obstacle);
                Toast.makeText(this, "Target " + targetId + " identified on Obstacle #" + obstacleNumber,
                        Toast.LENGTH_SHORT).show();
                Log.d(TAG, "Updated Obstacle #" + obstacleNumber + " with Target ID: " + targetId);
            } else {
                Log.w(TAG, "Obstacle #" + obstacleNumber + " not found");
            }
        } catch (JSONException e) {
            Log.e(TAG, "Failed to parse TARGET message: " + message, e);
        }
    }

    /**
     * Handle ROBOT message (C.10)
     * Format: {"cat": "location", "value": {"x": <x>, "y": <y>, "d": <direction>}}
     */
    private void handleRobotMessage(String message) {
        try {
            JSONObject json = new JSONObject(message);

            JSONObject value = json.getJSONObject("value");
            int x = value.getInt("x");
            int y = value.getInt("y");
            int d = value.getInt("d");

            // Convert numeric direction to Direction enum
            Robot.Direction direction =  Robot.Direction.fromNumeric(d);

            int gridSize = arenaMapView.getGridSize();
            if (x < 0 || x > gridSize - Robot.SIZE || y < 0 || y > gridSize - Robot.SIZE) {
                Log.w(TAG, "ROBOT coordinates out of bounds: (" + x + ", " + y + ")");
                return;
            }

            arenaMapView.updateRobotPosition(x, y, direction);
            positionText.setText(x + "," + y);
            directionText.setText(direction.name());

            Log.d(TAG, "Robot updated: position=(" + x + ", " + y + "), facing=" + direction);
        } catch (JSONException e) {
            Log.e(TAG, "Failed to parse ROBOT message: " + message, e);
        }
    }

    private Obstacle findObstacleById(int id) {
        for (Obstacle obs : arenaMapView.getObstacles()) {
            if (obs.getId() == id) {
                return obs;
            }
        }
        return null;
    }

    private void logMessage(String message, String colorHex) {
        String timestamp = new SimpleDateFormat("HH:mm:ss", Locale.getDefault()).format(new Date());
        String currentText = receivedText.getText().toString();

        if (currentText.equals("Waiting for data...")) {
            currentText = "";
        }

        String newText = "[" + timestamp + "] " + message + "\n" + currentText;

        String[] lines = newText.split("\n");
        if (lines.length > 20) {
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < 20; i++) {
                sb.append(lines[i]).append("\n");
            }
            newText = sb.toString();
        }

        receivedText.setText(newText);
    }

    private void clearMessages() {
        receivedText.setText("Waiting for data...");
        Toast.makeText(this, "Messages cleared", Toast.LENGTH_SHORT).show();
    }

    private void updateConnectionStatus(boolean connected, String deviceName) {
        isConnected = connected;
        connectedDeviceName = connected ? deviceName : null;

        updateActionBarMenuItem();

        if (connected) {
            upButton.setEnabled(true);
            downButton.setEnabled(true);
            leftButton.setEnabled(true);
            rightButton.setEnabled(true);
            tiltControlSwitch.setEnabled(true);
            sendButton.setEnabled(true);
        } else {
            upButton.setEnabled(false);
            downButton.setEnabled(false);
            leftButton.setEnabled(false);
            rightButton.setEnabled(false);
            tiltControlSwitch.setEnabled(false);
            tiltControlSwitch.setChecked(false);
            sendButton.setEnabled(false);
            disableTiltControl();
        }
    }

    private void handleIncomingConnection() {
        String deviceName = "Device";

        try {
            if (bluetoothService.socket != null && bluetoothService.socket.getRemoteDevice() != null) {
                deviceName = bluetoothService.socket.getRemoteDevice().getName();
                if (deviceName == null || deviceName.isEmpty()) {
                    deviceName = bluetoothService.socket.getRemoteDevice().getAddress();
                }
            }
        } catch (SecurityException e) {
            Log.e(TAG, "Permission denied getting device name", e);
            deviceName = "Connected Device";
        }

        updateConnectionStatus(true, deviceName);
        logMessage("Device connected!", "#4CAF50");
        Toast.makeText(this, "Device connected!", Toast.LENGTH_SHORT).show();
    }

    private void handleDisconnection() {
        updateConnectionStatus(false, "Disconnected");
        logMessage("Connection lost. Reconnecting...", "#F44336");
        Toast.makeText(this, "Device disconnected", Toast.LENGTH_SHORT).show();

        bluetoothService.restartServer();
    }

    private void checkPermissions() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.BLUETOOTH_CONNECT)
                    != PackageManager.PERMISSION_GRANTED) {
                ActivityCompat.requestPermissions(this,
                        new String[]{
                                Manifest.permission.BLUETOOTH_CONNECT,
                                Manifest.permission.BLUETOOTH_SCAN
                        },
                        REQUEST_BLUETOOTH_PERMISSIONS);
            }
        }
    }

    private void checkPermissionsAndConnect() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.BLUETOOTH_CONNECT)
                    != PackageManager.PERMISSION_GRANTED) {
                ActivityCompat.requestPermissions(this,
                        new String[]{Manifest.permission.BLUETOOTH_CONNECT},
                        REQUEST_BLUETOOTH_PERMISSIONS);
                return;
            }
        }
        showDeviceList();
    }

    private void showDeviceList() {
        try {
            Set<BluetoothDevice> pairedDevices = bluetoothAdapter.getBondedDevices();

            if (pairedDevices.isEmpty()) {
                Toast.makeText(this, "No paired devices found. Please pair your device first.",
                        Toast.LENGTH_LONG).show();
                return;
            }

            final ArrayList<BluetoothDevice> deviceList = new ArrayList<>(pairedDevices);
            String[] deviceNames = new String[deviceList.size()];

            for (int i = 0; i < deviceList.size(); i++) {
                deviceNames[i] = deviceList.get(i).getName() + "\n" + deviceList.get(i).getAddress();
            }

            AlertDialog.Builder builder = new AlertDialog.Builder(this);
            builder.setTitle("Select Bluetooth Device")
                    .setItems(deviceNames, (dialog, which) -> {
                        connectToDevice(deviceList.get(which));
                    })
                    .setNegativeButton("Cancel", null)
                    .show();

        } catch (SecurityException e) {
            Log.e(TAG, "Permission denied", e);
            Toast.makeText(this, "Bluetooth permission required", Toast.LENGTH_SHORT).show();
        }
    }

    private void connectToDevice(final BluetoothDevice device) {
        updateConnectionStatus(false, "Connecting...");

        new Thread(() -> {
            try {
                BluetoothSocket socket = device.createRfcommSocketToServiceRecord(MY_UUID);
                bluetoothAdapter.cancelDiscovery();
                socket.connect();

                runOnUiThread(() -> {
                    bluetoothService.connect(socket);
                    updateConnectionStatus(true, device.getName());
                    logMessage("Connected to " + device.getName(), "#4CAF50");
                    Toast.makeText(MainActivity.this, "Connected!", Toast.LENGTH_SHORT).show();
                });

            } catch (IOException e) {
                Log.e(TAG, "Connection failed", e);
                runOnUiThread(() -> {
                    updateConnectionStatus(false, "Connection Failed");
                    logMessage("Connection failed: " + e.getMessage(), "#F44336");
                    Toast.makeText(MainActivity.this, "Connection failed", Toast.LENGTH_LONG).show();
                });
            } catch (SecurityException e) {
                Log.e(TAG, "Permission denied", e);
                runOnUiThread(() -> {
                    updateConnectionStatus(false, "Permission Denied");
                });
            }
        }).start();
    }

    private void disconnect() {
        bluetoothService.stop();
        updateConnectionStatus(false, "Disconnected");
        logMessage("Disconnected", "#FF9800");
        bluetoothService.restartServer();
    }

    private void sendCommand(String command) {
        if (isConnected) {
            bluetoothService.write(command);
        } else {
            Log.d(TAG, "Cannot send - not connected: " + command);
        }
    }

    // ============================================================
    // TILT CONTROL (C.3)
    // ============================================================

    private void enableTiltControl() {
        if (accelerometer != null && isConnected) {
            isTiltControlEnabled = true;
            sensorManager.registerListener(this, accelerometer, SensorManager.SENSOR_DELAY_NORMAL);
            logMessage("Tilt control enabled", "#4CAF50");

            upButton.setEnabled(false);
            downButton.setEnabled(false);
            leftButton.setEnabled(false);
            rightButton.setEnabled(false);

        } else if (!isConnected) {
            tiltControlSwitch.setChecked(false);
            Toast.makeText(this, "Connect to a device first", Toast.LENGTH_SHORT).show();
        }
    }

    private void disableTiltControl() {
        if (isTiltControlEnabled) {
            isTiltControlEnabled = false;
            sensorManager.unregisterListener(this);

            if (isConnected) {
                upButton.setEnabled(true);
                downButton.setEnabled(true);
                leftButton.setEnabled(true);
                rightButton.setEnabled(true);
            }
        }
    }

    @Override
    public void onSensorChanged(SensorEvent event) {
        if (!isTiltControlEnabled || !isConnected) {
            return;
        }

        long currentTime = System.currentTimeMillis();

        if (currentTime - lastTiltCommandTime < TILT_COMMAND_INTERVAL) {
            return;
        }

        float x = event.values[0];
        float y = event.values[1];

        String command = null;

        if (Math.abs(y) > Math.abs(x)) {
            if (y < -TILT_THRESHOLD) {
                command = "move:up";
            } else if (y > TILT_THRESHOLD) {
                command = "move:down";
            }
        } else {
            if (x > TILT_THRESHOLD) {
                command = "move:left";
            } else if (x < -TILT_THRESHOLD) {
                command = "move:right";
            }
        }

        if (command != null) {
            sendCommand(command);
            lastTiltCommandTime = currentTime;
        }
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int accuracy) {
        // Not needed
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions,
                                           @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);

        if (requestCode == REQUEST_BLUETOOTH_PERMISSIONS) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                Toast.makeText(this, "Bluetooth permission granted", Toast.LENGTH_SHORT).show();
            } else {
                Toast.makeText(this, "Bluetooth permission is required for this app",
                        Toast.LENGTH_LONG).show();
            }
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (bluetoothService != null) {
            bluetoothService.stop();
        }
        disableTiltControl();
    }
}
