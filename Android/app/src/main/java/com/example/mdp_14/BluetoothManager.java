package com.example.mdp_14;

import android.Manifest;
import android.bluetooth.BluetoothAdapter;
import android.bluetooth.BluetoothDevice;
import android.bluetooth.BluetoothSocket;
import android.content.Context;
import android.content.pm.PackageManager;
import android.os.Build;
import android.os.Handler;
import android.os.Looper;
import android.os.Message;
import android.util.Log;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AlertDialog;
import androidx.core.content.ContextCompat;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Set;
import java.util.UUID;

/**
 * Manages all Bluetooth connectivity operations including:
 * - Connection/disconnection
 * - Device selection
 * - Message sending/receiving
 * - Server listening mode
 */
public class BluetoothManager {
    private static final String TAG = "BluetoothManager";
    private static final UUID MY_UUID = UUID.fromString("00001101-0000-1000-8000-00805F9B34FB");

    private final Context context;
    private final BluetoothAdapter bluetoothAdapter;
    private BluetoothService bluetoothService;
    private boolean isConnected = false;
    private String connectedDeviceName = null;

    private BluetoothCallback callback;

    public interface BluetoothCallback {
        void onConnected(String deviceName);
        void onDisconnected();
        void onMessageReceived(String message);
        void onMessageSent(String message);
        void onConnectionFailed(String error);
    }

    public BluetoothManager(Context context, BluetoothAdapter bluetoothAdapter) {
        this.context = context;
        this.bluetoothAdapter = bluetoothAdapter;
        this.bluetoothService = new BluetoothService(messageHandler, bluetoothAdapter);
    }

    public void setCallback(BluetoothCallback callback) {
        this.callback = callback;
    }

    // ============================================================
    // CONNECTION MANAGEMENT
    // ============================================================

    public void startListening() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            if (ContextCompat.checkSelfPermission(context, Manifest.permission.BLUETOOTH_CONNECT)
                    != PackageManager.PERMISSION_GRANTED) {
                Log.d(TAG, "Bluetooth permission not granted yet");
                return;
            }
        }

        if (!isConnected) {
            bluetoothService.startServer();
            Log.d(TAG, "Started listening for incoming connections");
        }
    }

    public void showDeviceSelectionDialog() {
        try {
            Set<BluetoothDevice> pairedDevices = bluetoothAdapter.getBondedDevices();

            if (pairedDevices.isEmpty()) {
                Toast.makeText(context, "No paired devices found. Please pair your device first.",
                        Toast.LENGTH_LONG).show();
                return;
            }

            final ArrayList<BluetoothDevice> deviceList = new ArrayList<>(pairedDevices);
            String[] deviceNames = new String[deviceList.size()];

            for (int i = 0; i < deviceList.size(); i++) {
                deviceNames[i] = deviceList.get(i).getName() + "\n" + deviceList.get(i).getAddress();
            }

            AlertDialog.Builder builder = new AlertDialog.Builder(context);
            builder.setTitle("Select Bluetooth Device")
                    .setItems(deviceNames, (dialog, which) -> {
                        connectToDevice(deviceList.get(which));
                    })
                    .setNegativeButton("Cancel", null)
                    .show();

        } catch (SecurityException e) {
            Log.e(TAG, "Permission denied", e);
            Toast.makeText(context, "Bluetooth permission required", Toast.LENGTH_SHORT).show();
        }
    }

    private void connectToDevice(final BluetoothDevice device) {
        new Thread(() -> {
            try {
                BluetoothSocket socket = device.createRfcommSocketToServiceRecord(MY_UUID);
                bluetoothAdapter.cancelDiscovery();
                socket.connect();

                bluetoothService.connect(socket);
                handleConnection(device.getName());

            } catch (IOException e) {
                Log.e(TAG, "Connection failed", e);
                if (callback != null) {
                    new Handler(Looper.getMainLooper()).post(() ->
                            callback.onConnectionFailed(e.getMessage())
                    );
                }
            } catch (SecurityException e) {
                Log.e(TAG, "Permission denied", e);
                if (callback != null) {
                    new Handler(Looper.getMainLooper()).post(() ->
                            callback.onConnectionFailed("Permission denied")
                    );
                }
            }
        }).start();
    }

    public void disconnect() {
        bluetoothService.stop();
        isConnected = false;
        connectedDeviceName = null;

        if (callback != null) {
            callback.onDisconnected();
        }

        bluetoothService.restartServer();
    }

    public void sendMessage(String message) {
        if (isConnected) {
            bluetoothService.write(message);
        } else {
            Log.w(TAG, "Cannot send - not connected: " + message);
        }
    }

    public void cleanup() {
        if (bluetoothService != null) {
            bluetoothService.stop();
        }
    }

    // ============================================================
    // GETTERS
    // ============================================================

    public boolean isConnected() {
        return isConnected;
    }

    public String getConnectedDeviceName() {
        return connectedDeviceName;
    }

    // ============================================================
    // MESSAGE HANDLER
    // ============================================================

    private final Handler messageHandler = new Handler(Looper.getMainLooper()) {
        @Override
        public void handleMessage(@NonNull Message msg) {
            switch (msg.what) {
                case BluetoothService.MESSAGE_READ:
                    String receivedMessage = (String) msg.obj;
                    if (callback != null) {
                        callback.onMessageReceived(receivedMessage);
                    }
                    break;

                case BluetoothService.MESSAGE_WRITE:
                    String sentMessage = (String) msg.obj;
                    if (callback != null) {
                        callback.onMessageSent(sentMessage);
                    }
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

    private void handleConnection(String deviceName) {
        isConnected = true;
        connectedDeviceName = deviceName;

        if (callback != null) {
            new Handler(Looper.getMainLooper()).post(() ->
                    callback.onConnected(deviceName)
            );
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

        isConnected = true;
        connectedDeviceName = deviceName;

        if (callback != null) {
            callback.onConnected(deviceName);
        }
    }

    private void handleDisconnection() {
        isConnected = false;
        connectedDeviceName = null;

        if (callback != null) {
            callback.onDisconnected();
        }

        bluetoothService.restartServer();
    }
}