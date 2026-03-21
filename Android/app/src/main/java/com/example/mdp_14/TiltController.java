package com.example.mdp_14;

import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorManager;
import android.widget.Toast;

/**
 * Manages tilt control functionality using device accelerometer
 */
public class TiltController {
    private static final String TAG = "TiltController";
    private static final long TILT_COMMAND_INTERVAL = 500; // ms
    private static final float TILT_THRESHOLD = 3.0f;

    private final SensorManager sensorManager;
    private final Sensor accelerometer;
    private final RobotController.CommandSender commandSender;

    private boolean isTiltControlEnabled = false;
    private long lastTiltCommandTime = 0;

    public TiltController(SensorManager sensorManager,
                          Sensor accelerometer,
                          RobotController.CommandSender commandSender) {
        this.sensorManager = sensorManager;
        this.accelerometer = accelerometer;
        this.commandSender = commandSender;
    }

    /**
     * Enable tilt control - register sensor listener
     */
    public void enable() {
        if (accelerometer != null) {
            isTiltControlEnabled = true;
            sensorManager.registerListener(
                    (android.hardware.SensorEventListener) commandSender,
                    accelerometer,
                    SensorManager.SENSOR_DELAY_NORMAL
            );
        }
    }

    /**
     * Disable tilt control - unregister sensor listener
     */
    public void disable() {
        if (isTiltControlEnabled) {
            isTiltControlEnabled = false;
            sensorManager.unregisterListener((android.hardware.SensorEventListener) commandSender);
        }
    }

    /**
     * Process sensor data and send tilt commands
     */
    public void onSensorChanged(SensorEvent event) {
        if (!isTiltControlEnabled) {
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
            commandSender.sendCommand(command);
            lastTiltCommandTime = currentTime;
        }
    }

    public boolean isEnabled() {
        return isTiltControlEnabled;
    }
}