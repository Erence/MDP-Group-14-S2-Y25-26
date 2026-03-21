package com.example.mdp_14;

import android.content.Context;
import android.content.SharedPreferences;
import android.content.res.ColorStateList;
import android.graphics.Color;
import android.widget.Button;
import android.widget.TextView;

import androidx.core.content.ContextCompat;

/**
 * Manages UI state and updates including:
 * - Button color states
 * - Status text updates
 * - Theme management
 */
public class UIManager {
    private static final String TAG = "UIManager";

    private final Context context;
    private final Button exploreButton;
    private final Button fastestPathButton;
    private final Button deleteObstacleButton;
    private final Button clearAllButton;
    private final Button resetButton;
    private final Button connectButton;

    private final TextView robotStatusText;
    private final TextView positionText;
    private final TextView directionText;

    public UIManager(Context context,
                     Button exploreButton,
                     Button fastestPathButton,
                     Button deleteObstacleButton,
                     Button clearAllButton,
                     Button resetButton,
                     Button connectButton,
                     TextView robotStatusText,
                     TextView positionText,
                     TextView directionText) {
        this.context = context;
        this.exploreButton = exploreButton;
        this.fastestPathButton = fastestPathButton;
        this.deleteObstacleButton = deleteObstacleButton;
        this.clearAllButton = clearAllButton;
        this.resetButton = resetButton;
        this.connectButton = connectButton;
        this.robotStatusText = robotStatusText;
        this.positionText = positionText;
        this.directionText = directionText;
    }

    // ============================================================
    // BUTTON STATE MANAGEMENT
    // ============================================================

    public void setExploreButtonActive() {
        exploreButton.setBackground(ContextCompat.getDrawable(context, R.drawable.bg_action_mint_pressed));
        exploreButton.setTextColor(ContextCompat.getColor(context, R.color.gold));
    }

    public void setFastestPathButtonActive() {
        fastestPathButton.setBackground(ContextCompat.getDrawable(context, R.drawable.bg_action_mint_pressed));
        fastestPathButton.setTextColor(ContextCompat.getColor(context, R.color.gold));
    }

    public void resetExploreButton() {
        exploreButton.setBackground(ContextCompat.getDrawable(context, R.drawable.bg_action_mint));
        exploreButton.setTextColor(ContextCompat.getColor(context, R.color.mint));
    }

    public void resetFastestPathButton() {
        fastestPathButton.setBackground(ContextCompat.getDrawable(context, R.drawable.bg_action_mint));
        fastestPathButton.setTextColor(ContextCompat.getColor(context, R.color.mint));
    }

    public void resetAllButtons() {
        resetExploreButton();
        resetFastestPathButton();
    }

    // ============================================================
    // STATUS UPDATES
    // ============================================================

    public void updateRobotStatus(String status) {
        robotStatusText.setText(status);
    }

    public void updateRobotPosition(int x, int y) {
        positionText.setText(x + "," + y);
    }

    public void updateRobotDirection(String direction) {
        directionText.setText(direction);
    }

    public void clearRobotPosition() {
        positionText.setText("-");
        directionText.setText("-");
    }

    public void updateRobotInfo(Robot robot) {
        if (robot != null) {
            updateRobotPosition(robot.getGridX(), robot.getGridY());
            updateRobotDirection(robot.getFacing().name());
        } else {
            clearRobotPosition();
        }
    }

    // ============================================================
    // CONNECTION STATE
    // ============================================================

    public void setConnectedState(String deviceName) {
        connectButton.setText(R.string.btn_disconnect);
        connectButton.setBackgroundTintList(ColorStateList.valueOf(Color.parseColor("#F44336"))); // Red
    }

    public void setDisconnectedState() {
        connectButton.setText(R.string.btn_connect);
        connectButton.setBackgroundTintList(ColorStateList.valueOf(Color.parseColor("#4CAF50"))); // Green
    }

    // ============================================================
    // THEME MANAGEMENT
    // ============================================================

    public void applyColorBlindMode() {
        SharedPreferences prefs = context.getSharedPreferences("settings", Context.MODE_PRIVATE);
        boolean cbMode = prefs.getBoolean("colour_blind_mode", false);

        if (cbMode) {
            applyColorBlindColors();
        } else {
            applyNormalColors();
        }
    }

    private void applyColorBlindColors() {
        exploreButton.setTextColor(ContextCompat.getColor(context, R.color.cb_mint));
        exploreButton.setBackground(ContextCompat.getDrawable(context, R.drawable.bg_action_cb_mint));

        fastestPathButton.setTextColor(ContextCompat.getColor(context, R.color.cb_mint));
        fastestPathButton.setBackground(ContextCompat.getDrawable(context, R.drawable.bg_action_cb_mint));

        deleteObstacleButton.setTextColor(ContextCompat.getColor(context, R.color.cb_coral));
        clearAllButton.setTextColor(ContextCompat.getColor(context, R.color.cb_coral));
        resetButton.setTextColor(ContextCompat.getColor(context, R.color.cb_coral));
    }

    private void applyNormalColors() {
        exploreButton.setTextColor(ContextCompat.getColor(context, R.color.mint));
        exploreButton.setBackground(ContextCompat.getDrawable(context, R.drawable.bg_action_mint));

        fastestPathButton.setTextColor(ContextCompat.getColor(context, R.color.mint));
        fastestPathButton.setBackground(ContextCompat.getDrawable(context, R.drawable.bg_action_mint));

        deleteObstacleButton.setTextColor(ContextCompat.getColor(context, R.color.coral));
        clearAllButton.setTextColor(ContextCompat.getColor(context, R.color.coral));
        resetButton.setTextColor(ContextCompat.getColor(context, R.color.coral));

        if (connectButton != null) {
            connectButton.setBackgroundTintList(ColorStateList.valueOf(
                    ContextCompat.getColor(context, R.color.mint)));
        }
    }
}