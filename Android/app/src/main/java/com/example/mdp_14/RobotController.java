package com.example.mdp_14;

import android.os.Handler;
import android.widget.TextView;

/**
 * Manages robot control operations including:
 * - Start/stop commands
 * - Timer management
 * - Status tracking
 */
public class RobotController {
    private static final String TAG = "RobotController";
    private static final long MAX_TIME_MILLIS = (5 * 60 + 55) * 1000; // 5:55

    private final TextView timerText;
    private final Handler timerHandler;
    private final CommandSender commandSender;

    private long startTime = 0;
    private boolean isTimerRunning = false;

    private RobotControllerCallback callback;

    public interface CommandSender {
        void sendCommand(String command);
    }

    public interface RobotControllerCallback {
        void onTimerExpired();
    }

    public RobotController(TextView timerText, CommandSender commandSender) {
        this.timerText = timerText;
        this.commandSender = commandSender;
        this.timerHandler = new Handler();
    }

    public void setCallback(RobotControllerCallback callback) {
        this.callback = callback;
    }

    // ============================================================
    // ROBOT CONTROL
    // ============================================================

    public void start() {
        if (!isTimerRunning) {
            startTimer();
        }
        commandSender.sendCommand("{\"cat\": \"control\", \"value\": \"start\"}");
    }

    public void stop() {
        stopTimer();
        commandSender.sendCommand("{\"cat\": \"control\", \"value\": \"stop\"}");
    }

    // ============================================================
    // TIMER MANAGEMENT
    // ============================================================

    private void startTimer() {
        startTime = System.currentTimeMillis();
        isTimerRunning = true;
        timerHandler.postDelayed(timerRunnable, 0);
    }

    public void stopTimer() {
        timerHandler.removeCallbacks(timerRunnable);
        isTimerRunning = false;
    }

    public void resetTimer() {
        stopTimer();
        timerText.setText("00:00:00");
    }

    private final Runnable timerRunnable = new Runnable() {
        @Override
        public void run() {
            long elapsedMillis = System.currentTimeMillis() - startTime;

            // Check if time limit reached
            if (elapsedMillis >= MAX_TIME_MILLIS) {
                elapsedMillis = MAX_TIME_MILLIS;
                timerText.setText("05:55:00");

                stopTimer();
                commandSender.sendCommand("{\"cat\": \"control\", \"value\": \"stop\"}");

                if (callback != null) {
                    callback.onTimerExpired();
                }

                return;
            }

            // Update timer display
            int seconds = (int) (elapsedMillis / 1000);
            int minutes = seconds / 60;
            seconds = seconds % 60;
            int millis = (int) (elapsedMillis % 1000) / 10;

            timerText.setText(String.format("%02d:%02d:%02d", minutes, seconds, millis));
            timerHandler.postDelayed(this, 10);
        }
    };

    public void cleanup() {
        timerHandler.removeCallbacks(timerRunnable);
    }

    // ============================================================
    // GETTERS
    // ============================================================

    public boolean isTimerRunning() {
        return isTimerRunning;
    }
}