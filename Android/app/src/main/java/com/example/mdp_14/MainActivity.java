package com.example.mdp_14;

import android.Manifest;
import android.bluetooth.BluetoothAdapter;
import android.bluetooth.BluetoothDevice;
import android.bluetooth.BluetoothSocket;
import android.content.Context;
import android.content.pm.PackageManager;
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
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.Switch;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.Locale;
import java.util.Set;
import java.util.UUID;

public class MainActivity extends AppCompatActivity implements SensorEventListener {
    private static final String TAG = "MainActivity";
    private static final int REQUEST_BLUETOOTH_PERMISSIONS = 1;
    private static final UUID MY_UUID = UUID.fromString("00001101-0000-1000-8000-00805F9B34FB");

    // Menu items for ActionBar
    private MenuItem deviceNameMenuItem;
    private MenuItem connectionIconMenuItem;
    private String connectedDeviceName = null;

    // UI Elements
    private View connectionIndicator;
    private TextView statusText;
    private TextView receivedText;
    private EditText messageInput;
    private ImageButton sendButton;
    private Button clearMessagesButton;
    private Button upButton;
    private Button downButton;
    private Button leftButton;
    private Button rightButton;
    private Switch tiltControlSwitch;

    private BluetoothAdapter bluetoothAdapter;
    private BluetoothService bluetoothService;
    private boolean isConnected = false;

    // Tilt control variables
    private SensorManager sensorManager;
    private Sensor accelerometer;
    private boolean isTiltControlEnabled = false;
    private long lastTiltCommandTime = 0;
    private static final long TILT_COMMAND_INTERVAL = 500;
    private static final float TILT_THRESHOLD = 3.0f;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // Initialize UI elements
        statusText = findViewById(R.id.statusTxt);
        receivedText = findViewById(R.id.receivedText);
        messageInput = findViewById(R.id.messageInput);
        sendButton = findViewById(R.id.sendBtn);
        clearMessagesButton = findViewById(R.id.clearMessagesBtn);
        upButton = findViewById(R.id.upBtn);
        downButton = findViewById(R.id.downBtn);
        leftButton = findViewById(R.id.leftBtn);
        rightButton = findViewById(R.id.rightBtn);
        tiltControlSwitch = findViewById(R.id.tiltControlSwitch);

        // Initialize sensor manager
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

        // Check permissions on startup
        checkPermissions();

        // Auto-start listening for incoming connections
        startListeningOnStartup();
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        // Create connect/disconnect menu item
        getMenuInflater().inflate(R.menu.main_menu, menu);

        // Get references to menu items from the inflated menu
        deviceNameMenuItem = menu.findItem(R.id.deviceNameTxt);
        connectionIconMenuItem = menu.findItem(R.id.connectionIcon);

        updateActionBarMenuItem();
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        if (item.getItemId() == R.id.connectionIcon) {
            if (!isConnected) {
                checkPermissionsAndConnect();
            } else {
                disconnect();
            }
            return true;
        }
        return super.onOptionsItemSelected(item);
    }

    /**
     * Update menu item based on connection status
     */
    private void updateActionBarMenuItem() {
        if (deviceNameMenuItem != null && connectionIconMenuItem != null) {
            if (isConnected && connectedDeviceName != null) {
                // Show device name and disconnect icon
                deviceNameMenuItem.setTitle(connectedDeviceName);
                deviceNameMenuItem.setVisible(true);
                connectionIconMenuItem.setIcon(R.drawable.disconnect);
            } else {
                // Hide device name, show connect icon
                deviceNameMenuItem.setVisible(false);
                connectionIconMenuItem.setIcon(R.drawable.connect);
            }
        }
    }

    /**
     * Auto-start listening when app starts
     */
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
                updateConnectionStatus(false, "Ready (Listening...)");
                Log.d(TAG, "Started listening for incoming connections");
            }
        }, 500);
    }

    /**
     * Handler for messages from BluetoothService
     */
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
                    addDetailedMessage("Sent: " + sentMessage, "#1976D2");
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

    /**
     * Handle incoming Bluetooth messages
     */
    private void handleIncomingMessage(String message) {

        // Add to detailed message log
        addDetailedMessage("Received: " + message, "#388E3C");

        // Parse different message types
        if (message.startsWith("TARGET,")) {
            handleTargetMessage(message);
        } else if (message.startsWith("ROBOT,")) {
            handleRobotUpdate(message);
        }
    }

    /**
     * Handle TARGET message (C.9)
     */
    private void handleTargetMessage(String message) {
        // TODO: Parse and update obstacle with target ID
        addDetailedMessage("Target detected: " + message, "#FFA000");
    }

    /**
     * Handle ROBOT position update (C.10)
     */
    private void handleRobotUpdate(String message) {
        // TODO: Parse and update robot position on map
        addDetailedMessage("Robot update: " + message, "#1976D2");
    }

    /**
     * Add timestamped message to detailed log
     */
    private void addDetailedMessage(String message, String colorHex) {
        String timestamp = new SimpleDateFormat("HH:mm:ss", Locale.getDefault()).format(new Date());
        String currentText = receivedText.getText().toString();

        // Clear "Waiting for data..." on first message
        if (currentText.equals("Waiting for data...")) {
            currentText = "";
        }

        String newText = "[" + timestamp + "] " + message + "\n" + currentText;

        // Limit to last 20 lines
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

    /**
     * Clear message log
     */
    private void clearMessages() {
        receivedText.setText("Waiting for data...");
        Toast.makeText(this, "Messages cleared", Toast.LENGTH_SHORT).show();
    }

    /**
     * Update connection status UI
     */
    private void updateConnectionStatus(boolean connected, String deviceName) {
        isConnected = connected;
        connectedDeviceName = connected ? deviceName : null;

        // Update ActionBar
        updateActionBarMenuItem();

        if (connected) {
            // Enable controls
            upButton.setEnabled(true);
            downButton.setEnabled(true);
            leftButton.setEnabled(true);
            rightButton.setEnabled(true);
            tiltControlSwitch.setEnabled(true);
            sendButton.setEnabled(true);
        } else {
            // Disable controls
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

    /**
     * Handle incoming connection
     */
    private void handleIncomingConnection() {
        // For incoming connections, try to get device name from the socket
        String deviceName = "Device";

        // Try to get the actual device name if possible
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
        addDetailedMessage("Device connected!", "#4CAF50");
        Toast.makeText(this, "Device connected!", Toast.LENGTH_SHORT).show();
    }

    /**
     * Handle disconnection
     */
    private void handleDisconnection() {
        updateConnectionStatus(false, "Disconnected");
        addDetailedMessage("Connection lost. Reconnecting...", "#F44336");
        Toast.makeText(this, "Device disconnected", Toast.LENGTH_SHORT).show();

        // Restart listening after disconnect (C.8)
        bluetoothService.restartServer();
        updateConnectionStatus(false, "Ready (Listening...)");
    }

    /**
     * Check and request Bluetooth permissions
     */
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

    /**
     * Check permissions and initiate connection (C.2)
     */
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

    /**
     * Show list of paired devices (C.2)
     */
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

    /**
     * Connect to the selected device
     */
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
                    addDetailedMessage("Connected to " + device.getName(), "#4CAF50");
                    Toast.makeText(MainActivity.this, "Connected!", Toast.LENGTH_SHORT).show();
                });

            } catch (IOException e) {
                Log.e(TAG, "Connection failed", e);
                runOnUiThread(() -> {
                    updateConnectionStatus(false, "Connection Failed");
                    addDetailedMessage("Connection failed: " + e.getMessage(), "#F44336");
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

    /**
     * Disconnect from device
     */
    private void disconnect() {
        bluetoothService.stop();
        updateConnectionStatus(false, "Ready (Listening...)");
        addDetailedMessage("Disconnected", "#FF9800");

        // Restart listening after manual disconnect
        new Handler(Looper.getMainLooper()).postDelayed(() -> {
            bluetoothService.startServer();
        }, 1000);
    }

    /**
     * Send command via Bluetooth (C.1, C.3)
     */
    private void sendCommand(String command) {
        if (isConnected) {
            bluetoothService.write(command);
        } else {
            Toast.makeText(this, "Not connected to any device", Toast.LENGTH_SHORT).show();
        }
    }

    /**
     * Enable tilt control
     */
    private void enableTiltControl() {
        if (accelerometer != null && isConnected) {
            isTiltControlEnabled = true;
            sensorManager.registerListener(this, accelerometer, SensorManager.SENSOR_DELAY_NORMAL);
            addDetailedMessage("Tilt control enabled", "#4CAF50");
        } else if (!isConnected) {
            tiltControlSwitch.setChecked(false);
            Toast.makeText(this, "Connect to a device first", Toast.LENGTH_SHORT).show();
        }
    }

    /**
     * Disable tilt control
     */
    private void disableTiltControl() {
        if (isTiltControlEnabled) {
            isTiltControlEnabled = false;
            sensorManager.unregisterListener(this);
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