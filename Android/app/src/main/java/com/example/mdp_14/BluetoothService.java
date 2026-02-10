package com.example.mdp_14;

import android.bluetooth.BluetoothAdapter;
import android.bluetooth.BluetoothServerSocket;
import android.bluetooth.BluetoothSocket;
import android.os.Handler;
import android.util.Log;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.UUID;

public class BluetoothService {
    private static final String TAG = "BluetoothService";
    private static final String NAME = "BluetoothSerial";
    private static final UUID MY_UUID = UUID.fromString("00001101-0000-1000-8000-00805F9B34FB");

    private BluetoothAdapter bluetoothAdapter;
    public BluetoothSocket socket; // Made public to access device info
    private InputStream inputStream;
    private OutputStream outputStream;
    private ConnectedThread connectedThread;
    private AcceptThread acceptThread;
    private Handler handler;

    // Message types for handler
    public static final int MESSAGE_READ = 0;
    public static final int MESSAGE_WRITE = 1;
    public static final int MESSAGE_DISCONNECTED = 2;
    public static final int MESSAGE_CONNECTED = 3;

    public BluetoothService(Handler handler, BluetoothAdapter adapter) {
        this.handler = handler;
        this.bluetoothAdapter = adapter;
    }

    /**
     * Start listening for incoming connections (Server mode)
     * This runs continuously in the background
     */
    public synchronized void startServer() {
        Log.d(TAG, "startServer: Starting server mode");

        // Don't interrupt existing connection
        if (connectedThread != null) {
            Log.d(TAG, "Already connected, not starting server");
            return;
        }

        // Cancel existing accept thread if any
        if (acceptThread != null) {
            acceptThread.cancel();
            acceptThread = null;
        }

        // Start the thread to listen for incoming connections
        acceptThread = new AcceptThread();
        acceptThread.start();
    }

    /**
     * Start the connection with the given socket
     */
    public synchronized void connect(BluetoothSocket socket) {
        Log.d(TAG, "connect: Starting connection");

        // Cancel any existing threads
        if (connectedThread != null) {
            connectedThread.cancel();
            connectedThread = null;
        }

        if (acceptThread != null) {
            acceptThread.cancel();
            acceptThread = null;
        }

        this.socket = socket;

        try {
            inputStream = socket.getInputStream();
            outputStream = socket.getOutputStream();
        } catch (IOException e) {
            Log.e(TAG, "Error getting streams", e);
            return;
        }

        // Start the thread to manage the connection
        connectedThread = new ConnectedThread();
        connectedThread.start();
    }

    /**
     * Write to the connected device
     */
    public void write(String message) {
        if (connectedThread != null) {
            connectedThread.write(message.getBytes());
        }
    }

    /**
     * Stop all threads
     */
    public synchronized void stop() {
        Log.d(TAG, "stop");

        if (connectedThread != null) {
            connectedThread.cancel();
            connectedThread = null;
        }

        if (acceptThread != null) {
            acceptThread.cancel();
            acceptThread = null;
        }

        try {
            if (socket != null) {
                socket.close();
                socket = null;
            }
        } catch (IOException e) {
            Log.e(TAG, "Error closing socket", e);
        }
    }

    /**
     * Restart server after disconnection
     */
    public void restartServer() {
        Log.d(TAG, "restartServer: Restarting server mode");

        new Thread(() -> {
            synchronized (this) {
                // Stop everything first
                if (connectedThread != null) {
                    connectedThread.cancel();
                    connectedThread = null;
                }

                if (acceptThread != null) {
                    acceptThread.cancel();
                    acceptThread = null;
                }

                try {
                    if (socket != null) {
                        socket.close();
                        socket = null;
                    }
                } catch (IOException e) {
                    Log.e(TAG, "Error closing socket during restart", e);
                }
            }

            // Delay to let OS fully release the socket
            try {
                Thread.sleep(1000);
            } catch (InterruptedException e) {
                Log.e(TAG, "Interrupted during restart delay", e);
            }

            // Now start server again
            startServer();
            Log.d(TAG, "restartServer: Server restarted successfully");
        }).start();
    }

    /**
     * Thread to listen for incoming connections (Server)
     */
    private class AcceptThread extends Thread {
        private final BluetoothServerSocket serverSocket;

        public AcceptThread() {
            BluetoothServerSocket tmp = null;
            try {
                tmp = bluetoothAdapter.listenUsingRfcommWithServiceRecord(NAME, MY_UUID);
                Log.d(TAG, "AcceptThread: Server socket created, listening...");
            } catch (IOException e) {
                Log.e(TAG, "AcceptThread: Socket listen() failed", e);
            } catch (SecurityException e) {
                Log.e(TAG, "AcceptThread: Permission denied", e);
            }
            serverSocket = tmp;
        }

        public void run() {
            BluetoothSocket socket = null;

            // Keep listening until connected or exception occurs
            while (true) {
                try {
                    Log.d(TAG, "AcceptThread: Waiting for incoming connection...");
                    socket = serverSocket.accept(); // This blocks until connection

                    Log.d(TAG, "AcceptThread: Connection accepted!");

                    if (socket != null) {
                        // Connection accepted, start connected thread
                        synchronized (BluetoothService.this) {
                            connect(socket);
                            handler.obtainMessage(MESSAGE_CONNECTED).sendToTarget();
                        }

                        // Close server socket as we have a connection
                        serverSocket.close();
                        break;
                    }
                } catch (IOException e) {
                    Log.e(TAG, "AcceptThread: accept() failed", e);
                    break;
                }
            }
        }

        public void cancel() {
            try {
                if (serverSocket != null) {
                    serverSocket.close();
                }
            } catch (IOException e) {
                Log.e(TAG, "AcceptThread: close() failed", e);
            }
        }
    }

    /**
     * Thread to handle reading and writing data
     */
    private class ConnectedThread extends Thread {
        private final InputStream inStream;
        private final OutputStream outStream;
        private volatile boolean cancelled = false;

        public ConnectedThread() {
            inStream = inputStream;
            outStream = outputStream;
        }

        public void run() {
            byte[] buffer = new byte[1024];
            int bytes;

            // Keep listening to the InputStream while connected
            while (!cancelled) {
                try {
                    // Read from the InputStream
                    bytes = inStream.read(buffer);

                    if (bytes == -1) {
                        // Stream closed cleanly
                        Log.d(TAG, "Stream closed, disconnected");
                        handler.obtainMessage(MESSAGE_DISCONNECTED).sendToTarget();
                        break;
                    }

                    // Send the obtained bytes to the UI Activity
                    String receivedMessage = new String(buffer, 0, bytes).trim();
                    if (!receivedMessage.isEmpty()) {
                        handler.obtainMessage(MESSAGE_READ, bytes, -1, receivedMessage)
                                .sendToTarget();
                    }

                } catch (IOException e) {
                    if (!cancelled) {
                        Log.e(TAG, "disconnected", e);
                        handler.obtainMessage(MESSAGE_DISCONNECTED).sendToTarget();
                    }
                    break;
                }
            }
        }

        /**
         * Write to the connected OutStream
         */
        public void write(byte[] buffer) {
            try {
                outStream.write(buffer);
                outStream.flush(); // Flush to ensure data is sent

                // Share the sent message with UI
                String sentMessage = new String(buffer);
                handler.obtainMessage(MESSAGE_WRITE, -1, -1, sentMessage)
                        .sendToTarget();

            } catch (IOException e) {
                Log.e(TAG, "Error writing to stream", e);
            }
        }

        public void cancel() {
            cancelled = true;
            try {
                inStream.close();
                outStream.close();
            } catch (IOException e) {
                Log.e(TAG, "Error closing streams", e);
            }
        }
    }
}