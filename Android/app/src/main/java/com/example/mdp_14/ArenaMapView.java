package com.example.mdp_14;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Path;
import android.graphics.RectF;
import android.util.AttributeSet;
import android.view.GestureDetector;
import android.view.MotionEvent;
import android.view.View;

import java.util.ArrayList;
import java.util.List;

/**
 * Custom view for displaying the MDP arena map with draggable obstacles and robot.
 * Supports a 20x20 grid with obstacles that can be placed, moved, and edited.
 */
public class ArenaMapView extends View {
    private static final int GRID_SIZE = 20;

    private float cellSize;
    private float offsetX, offsetY;

    // Paints
    private Paint gridPaint;
    private Paint obstaclePaint;
    private Paint obstacleDeletePaint;  // For drag-to-delete visual feedback
    private Paint targetIndicatorPaint;
    private Paint targetTextPaint;
    private Paint gridLabelPaint;
    private Paint selectedPaint;
    private Paint robotPaint;
    private Paint robotDirectionPaint;

    // Data
    private List<Obstacle> obstacles = new ArrayList<>();
    private Robot robot = null;

    // Drag state
    private Obstacle draggedObstacle = null;
    private Obstacle selectedObstacle = null;
    private boolean isDraggingRobot = false;
    private float dragOffsetX, dragOffsetY;

    // Track dragged obstacle screen position for drag-to-delete
    private float draggedScreenX, draggedScreenY;
    private boolean isOutsideGrid = false;

    // Lock state
    private boolean isDragLocked = false;

    private OnObstacleActionListener listener;
    private GestureDetector gestureDetector;

    public interface OnObstacleActionListener {
        void onObstacleLongPress(Obstacle obstacle);
        void onObstacleSelected(Obstacle obstacle);
        void onObstaclePositionChanged(Obstacle obstacle);
        void onObstacleRemovedByDrag(Obstacle obstacle);
        void onRobotPositionChanged(Robot robot);
        void onEmptyCellTap(int gridX, int gridY);
    }

    public ArenaMapView(Context context) {
        super(context);
        init();
    }

    public ArenaMapView(Context context, AttributeSet attrs) {
        super(context, attrs);
        init();
    }

    public ArenaMapView(Context context, AttributeSet attrs, int defStyleAttr) {
        super(context, attrs, defStyleAttr);
        init();
    }

    private void init() {
        // Grid lines paint
        gridPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        gridPaint.setColor(Color.LTGRAY);
        gridPaint.setStrokeWidth(1f);
        gridPaint.setStyle(Paint.Style.STROKE);

        // Obstacle fill paint
        obstaclePaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        obstaclePaint.setColor(Color.BLACK);
        obstaclePaint.setStyle(Paint.Style.FILL);

        // Obstacle delete preview paint (red, semi-transparent)
        obstacleDeletePaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        obstacleDeletePaint.setColor(Color.parseColor("#80FF0000"));  // Semi-transparent red
        obstacleDeletePaint.setStyle(Paint.Style.FILL);

        // Target indicator paint
        targetIndicatorPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        targetIndicatorPaint.setColor(Color.RED);
        targetIndicatorPaint.setStyle(Paint.Style.FILL);

        // Target text paint
        targetTextPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        targetTextPaint.setColor(Color.WHITE);
        targetTextPaint.setTextAlign(Paint.Align.CENTER);

        // Grid label paint
        gridLabelPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        gridLabelPaint.setColor(Color.DKGRAY);
        gridLabelPaint.setTextSize(24f);
        gridLabelPaint.setTextAlign(Paint.Align.CENTER);

        // Selected highlight
        selectedPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        selectedPaint.setColor(Color.BLUE);
        selectedPaint.setStrokeWidth(4f);
        selectedPaint.setStyle(Paint.Style.STROKE);

        // Robot body paint
        robotPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        robotPaint.setColor(Color.parseColor("#4CAF50")); // Green
        robotPaint.setStyle(Paint.Style.FILL);

        // Robot direction indicator paint
        robotDirectionPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        robotDirectionPaint.setColor(Color.parseColor("#1B5E20")); // Dark green
        robotDirectionPaint.setStyle(Paint.Style.FILL);

        // Gesture detector for long press
        gestureDetector = new GestureDetector(getContext(), new GestureDetector.SimpleOnGestureListener() {
            @Override
            public void onLongPress(MotionEvent e) {
                if (isDragLocked) return;

                int[] gridPos = screenToGrid(e.getX(), e.getY());
                Obstacle obstacle = findObstacleAt(gridPos[0], gridPos[1]);
                if (obstacle != null && listener != null) {
                    selectedObstacle = obstacle;
                    invalidate();
                    listener.onObstacleLongPress(obstacle);
                }
            }

            @Override
            public boolean onSingleTapUp(MotionEvent e) {
                int[] gridPos = screenToGrid(e.getX(), e.getY());
                Obstacle obstacle = findObstacleAt(gridPos[0], gridPos[1]);
                if (obstacle != null) {
                    selectedObstacle = obstacle;
                    invalidate();
                    if (listener != null) {
                        listener.onObstacleSelected(obstacle);
                    }
                } else {
                    if (listener != null) {
                        listener.onEmptyCellTap(gridPos[0], gridPos[1]);
                    }
                }
                return true;
            }
        });
    }

    @Override
    protected void onSizeChanged(int w, int h, int oldw, int oldh) {
        super.onSizeChanged(w, h, oldw, oldh);
        calculateDimensions();
    }

    private void calculateDimensions() {
        int width = getWidth();
        int height = getHeight();

        float labelMarginLeft = 30f;   // For row labels on left
        float labelMarginBottom = 30f; // For column labels at bottom
        float availableWidth = width - labelMarginLeft;
        float availableHeight = height - labelMarginBottom;

        cellSize = Math.min(availableWidth / GRID_SIZE, availableHeight / GRID_SIZE);

        float gridWidth = cellSize * GRID_SIZE;
        float gridHeight = cellSize * GRID_SIZE;
        offsetX = labelMarginLeft + (availableWidth - gridWidth) / 2;
        offsetY = (availableHeight - gridHeight) / 2;

        targetTextPaint.setTextSize(cellSize * 0.5f);
        gridLabelPaint.setTextSize(cellSize * 0.4f);
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);

        canvas.drawColor(Color.WHITE);
        drawGrid(canvas);

        // Draw obstacles (skip the one being dragged, we'll draw it separately)
        for (Obstacle obstacle : obstacles) {
            if (obstacle != draggedObstacle) {
                drawObstacle(canvas, obstacle, obstacle == selectedObstacle);
            }
        }

        // Draw dragged obstacle at its current drag position
        if (draggedObstacle != null) {
            drawDraggedObstacle(canvas, draggedObstacle);
        }

        // Draw robot on top
        if (robot != null) {
            drawRobot(canvas);
        }
    }

    private void drawGrid(Canvas canvas) {
        // Draw vertical lines
        for (int i = 0; i <= GRID_SIZE; i++) {
            float x = offsetX + i * cellSize;
            canvas.drawLine(x, offsetY, x, offsetY + GRID_SIZE * cellSize, gridPaint);
        }

        // Draw horizontal lines
        for (int i = 0; i <= GRID_SIZE; i++) {
            float y = offsetY + i * cellSize;
            canvas.drawLine(offsetX, y, offsetX + GRID_SIZE * cellSize, y, gridPaint);
        }

        // Draw column labels (0-19) at bottom
        for (int i = 0; i < GRID_SIZE; i++) {
            float x = offsetX + i * cellSize + cellSize / 2;
            canvas.drawText(String.valueOf(i), x, offsetY + GRID_SIZE * cellSize + gridLabelPaint.getTextSize() + 5, gridLabelPaint);
        }

        // Draw row labels (0-19) with 0 at bottom, 19 at top
        for (int i = 0; i < GRID_SIZE; i++) {
            int gridY = GRID_SIZE - 1 - i;  // Flip: screen row 0 = grid row 19
            float y = offsetY + i * cellSize + cellSize / 2 + gridLabelPaint.getTextSize() / 3;
            canvas.drawText(String.valueOf(gridY), offsetX - 15, y, gridLabelPaint);
        }
    }

    private void drawObstacle(Canvas canvas, Obstacle obstacle, boolean isSelected) {
        float left = offsetX + obstacle.getGridX() * cellSize;
        // Flip Y: gridY=0 at bottom, so higher gridY = lower screen Y
        float top = offsetY + (GRID_SIZE - obstacle.getGridY() - obstacle.getHeight()) * cellSize;
        float right = left + obstacle.getWidth() * cellSize;
        float bottom = top + obstacle.getHeight() * cellSize;

        RectF rect = new RectF(left + 2, top + 2, right - 2, bottom - 2);
        canvas.drawRect(rect, obstaclePaint);

        if (isSelected) {
            canvas.drawRect(rect, selectedPaint);
        }

        // Always draw target face indicator (red bar showing which face has the target image)
        drawTargetFaceIndicator(canvas, obstacle, left, top, right, bottom);

        // Draw obstacle ID (small white number)
        float centerX = (left + right) / 2;
        float centerY = (top + bottom) / 2;

        if (obstacle.hasRecognizedTarget()) {
            // If target has been recognized, show ID small at top and recognized target large in center
            Paint smallIdPaint = new Paint(targetTextPaint);
            smallIdPaint.setTextSize(cellSize * 0.3f);
            canvas.drawText(String.valueOf(obstacle.getId()), centerX, top + cellSize * 0.4f, smallIdPaint);

            // Draw recognized target ID in large white font
            Paint largeTargetPaint = new Paint(targetTextPaint);
            largeTargetPaint.setTextSize(cellSize * 0.7f);
            largeTargetPaint.setFakeBoldText(true);
            canvas.drawText(obstacle.getRecognizedTargetId(), centerX, centerY + largeTargetPaint.getTextSize() / 3, largeTargetPaint);
        } else {
            // No recognized target yet, just show obstacle ID in center (smaller font)
            Paint smallIdPaint = new Paint(targetTextPaint);
            smallIdPaint.setTextSize(cellSize * 0.35f);
            canvas.drawText(String.valueOf(obstacle.getId()), centerX, centerY + smallIdPaint.getTextSize() / 3, smallIdPaint);
        }
    }

    /**
     * Draw obstacle being dragged at its current screen position with delete feedback
     */
    private void drawDraggedObstacle(Canvas canvas, Obstacle obstacle) {
        float left = draggedScreenX;
        float top = draggedScreenY;
        float right = left + obstacle.getWidth() * cellSize;
        float bottom = top + obstacle.getHeight() * cellSize;

        RectF rect = new RectF(left + 2, top + 2, right - 2, bottom - 2);

        // Use delete paint if outside grid, otherwise normal paint
        if (isOutsideGrid) {
            canvas.drawRect(rect, obstacleDeletePaint);
        } else {
            canvas.drawRect(rect, obstaclePaint);
            canvas.drawRect(rect, selectedPaint);  // Always show selection when dragging
        }

        // Draw target face indicator
        drawTargetFaceIndicator(canvas, obstacle, left, top, right, bottom);

        // Draw obstacle ID (smaller font)
        float centerX = (left + right) / 2;
        float centerY = (top + bottom) / 2;
        Paint smallIdPaint = new Paint(targetTextPaint);
        smallIdPaint.setTextSize(cellSize * 0.35f);
        canvas.drawText(String.valueOf(obstacle.getId()), centerX, centerY + smallIdPaint.getTextSize() / 3, smallIdPaint);
    }

    /**
     * Draw the red indicator bar showing which face has the target image
     */
    private void drawTargetFaceIndicator(Canvas canvas, Obstacle obstacle,
                                          float left, float top, float right, float bottom) {
        float indicatorThickness = cellSize * 0.15f;
        RectF indicatorRect;

        switch (obstacle.getTargetFace()) {
            case NORTH:
                indicatorRect = new RectF(left + 2, top + 2, right - 2, top + indicatorThickness);
                break;
            case SOUTH:
                indicatorRect = new RectF(left + 2, bottom - indicatorThickness, right - 2, bottom - 2);
                break;
            case EAST:
                indicatorRect = new RectF(right - indicatorThickness, top + 2, right - 2, bottom - 2);
                break;
            case WEST:
                indicatorRect = new RectF(left + 2, top + 2, left + indicatorThickness, bottom - 2);
                break;
            default:
                return;
        }

        canvas.drawRect(indicatorRect, targetIndicatorPaint);
    }

    private void drawRobot(Canvas canvas) {
        float left = offsetX + robot.getGridX() * cellSize;
        // Flip Y: gridY=0 at bottom
        float top = offsetY + (GRID_SIZE - robot.getGridY() - Robot.SIZE) * cellSize;
        float right = left + Robot.SIZE * cellSize;
        float bottom = top + Robot.SIZE * cellSize;

        // Draw robot body (green square)
        RectF rect = new RectF(left + 3, top + 3, right - 3, bottom - 3);
        canvas.drawRect(rect, robotPaint);

        // Draw direction triangle
        float centerX = (left + right) / 2;
        float centerY = (top + bottom) / 2;
        float triangleSize = cellSize * 0.6f;

        Path triangle = new Path();
        switch (robot.getFacing()) {
            case NORTH:
                triangle.moveTo(centerX, top + 6);                          // Top point
                triangle.lineTo(centerX - triangleSize / 2, centerY);       // Bottom left
                triangle.lineTo(centerX + triangleSize / 2, centerY);       // Bottom right
                break;
            case SOUTH:
                triangle.moveTo(centerX, bottom - 6);                       // Bottom point
                triangle.lineTo(centerX - triangleSize / 2, centerY);       // Top left
                triangle.lineTo(centerX + triangleSize / 2, centerY);       // Top right
                break;
            case EAST:
                triangle.moveTo(right - 6, centerY);                        // Right point
                triangle.lineTo(centerX, centerY - triangleSize / 2);       // Top left
                triangle.lineTo(centerX, centerY + triangleSize / 2);       // Bottom left
                break;
            case WEST:
                triangle.moveTo(left + 6, centerY);                         // Left point
                triangle.lineTo(centerX, centerY - triangleSize / 2);       // Top right
                triangle.lineTo(centerX, centerY + triangleSize / 2);       // Bottom right
                break;
        }
        triangle.close();
        canvas.drawPath(triangle, robotDirectionPaint);
    }

    @Override
    public boolean onTouchEvent(MotionEvent event) {
        gestureDetector.onTouchEvent(event);

        // If drag is locked, only allow tap/selection, not dragging
        if (isDragLocked) {
            return true;
        }

        int[] gridPos = screenToGrid(event.getX(), event.getY());
        int gridX = gridPos[0];
        int gridY = gridPos[1];

        switch (event.getAction()) {
            case MotionEvent.ACTION_DOWN:
                // Check robot first (it's on top)
                if (robot != null && robot.containsPoint(gridX, gridY)) {
                    isDraggingRobot = true;
                    dragOffsetX = event.getX() - (offsetX + robot.getGridX() * cellSize);
                    // Flip Y for drag offset calculation
                    dragOffsetY = event.getY() - (offsetY + (GRID_SIZE - robot.getGridY() - Robot.SIZE) * cellSize);
                    return true;
                }

                // Then check obstacles
                Obstacle obstacle = findObstacleAt(gridX, gridY);
                if (obstacle != null) {
                    draggedObstacle = obstacle;
                    selectedObstacle = obstacle;
                    // Calculate drag offset from touch point to obstacle's top-left corner
                    draggedScreenX = offsetX + obstacle.getGridX() * cellSize;
                    draggedScreenY = offsetY + (GRID_SIZE - obstacle.getGridY() - obstacle.getHeight()) * cellSize;
                    dragOffsetX = event.getX() - draggedScreenX;
                    dragOffsetY = event.getY() - draggedScreenY;
                    isOutsideGrid = false;
                    invalidate();
                    return true;
                }
                break;

            case MotionEvent.ACTION_MOVE:
                if (isDraggingRobot && robot != null) {
                    float newX = event.getX() - dragOffsetX;
                    float newY = event.getY() - dragOffsetY;
                    int newGridX = Math.round((newX - offsetX) / cellSize);
                    // Flip Y: convert screen position to grid position
                    int newScreenGridY = Math.round((newY - offsetY) / cellSize);
                    int newGridY = GRID_SIZE - Robot.SIZE - newScreenGridY;

                    // Clamp to grid bounds (robot is 3x3)
                    newGridX = Math.max(0, Math.min(GRID_SIZE - Robot.SIZE, newGridX));
                    newGridY = Math.max(0, Math.min(GRID_SIZE - Robot.SIZE, newGridY));

                    if (newGridX != robot.getGridX() || newGridY != robot.getGridY()) {
                        robot.setGridX(newGridX);
                        robot.setGridY(newGridY);
                        invalidate();
                    }
                    return true;
                }

                if (draggedObstacle != null) {
                    // Track screen position for visual feedback
                    draggedScreenX = event.getX() - dragOffsetX;
                    draggedScreenY = event.getY() - dragOffsetY;

                    // Calculate grid position (unclamped to check if outside)
                    int newGridX = Math.round((draggedScreenX - offsetX) / cellSize);
                    int newScreenGridY = Math.round((draggedScreenY - offsetY) / cellSize);
                    int newGridY = GRID_SIZE - draggedObstacle.getHeight() - newScreenGridY;

                    // Check if obstacle is outside the grid (for delete feedback)
                    float gridLeft = offsetX;
                    float gridTop = offsetY;
                    float gridRight = offsetX + GRID_SIZE * cellSize;
                    float gridBottom = offsetY + GRID_SIZE * cellSize;
                    float obsCenterX = draggedScreenX + (draggedObstacle.getWidth() * cellSize) / 2;
                    float obsCenterY = draggedScreenY + (draggedObstacle.getHeight() * cellSize) / 2;

                    isOutsideGrid = obsCenterX < gridLeft || obsCenterX > gridRight ||
                                    obsCenterY < gridTop || obsCenterY > gridBottom;

                    // Update grid position only if inside grid bounds
                    if (!isOutsideGrid) {
                        newGridX = Math.max(0, Math.min(GRID_SIZE - draggedObstacle.getWidth(), newGridX));
                        newGridY = Math.max(0, Math.min(GRID_SIZE - draggedObstacle.getHeight(), newGridY));
                        draggedObstacle.setGridX(newGridX);
                        draggedObstacle.setGridY(newGridY);
                    }

                    invalidate();
                    return true;
                }
                break;

            case MotionEvent.ACTION_UP:
            case MotionEvent.ACTION_CANCEL:
                if (isDraggingRobot && robot != null && listener != null) {
                    listener.onRobotPositionChanged(robot);
                }
                isDraggingRobot = false;

                if (draggedObstacle != null) {
                    if (isOutsideGrid) {
                        // Remove the obstacle if dropped outside the grid
                        Obstacle removedObstacle = draggedObstacle;
                        obstacles.remove(draggedObstacle);
                        if (selectedObstacle == draggedObstacle) {
                            selectedObstacle = null;
                        }
                        if (listener != null) {
                            listener.onObstacleRemovedByDrag(removedObstacle);
                        }
                    } else {
                        if (listener != null) {
                            listener.onObstaclePositionChanged(draggedObstacle);
                        }
                    }
                    draggedObstacle = null;
                    isOutsideGrid = false;
                    invalidate();
                }
                break;
        }

        return true;
    }

    private int[] screenToGrid(float screenX, float screenY) {
        int gridX = (int) ((screenX - offsetX) / cellSize);
        int screenGridY = (int) ((screenY - offsetY) / cellSize);
        // Flip Y: screen row 0 (top) = grid row 19, screen row 19 (bottom) = grid row 0
        int gridY = GRID_SIZE - 1 - screenGridY;
        gridX = Math.max(0, Math.min(GRID_SIZE - 1, gridX));
        gridY = Math.max(0, Math.min(GRID_SIZE - 1, gridY));
        return new int[]{gridX, gridY};
    }

    private Obstacle findObstacleAt(int gridX, int gridY) {
        for (Obstacle obstacle : obstacles) {
            if (obstacle.containsPoint(gridX, gridY)) {
                return obstacle;
            }
        }
        return null;
    }

    // Public API methods

    public void setOnObstacleActionListener(OnObstacleActionListener listener) {
        this.listener = listener;
    }

    public void addObstacle(Obstacle obstacle) {
        obstacles.add(obstacle);
        invalidate();
    }

    public void addObstacle(int gridX, int gridY) {
        addObstacle(new Obstacle(gridX, gridY));
    }

    public void addObstacle(int gridX, int gridY, int width, int height) {
        addObstacle(new Obstacle(gridX, gridY, width, height));
    }

    public void removeObstacle(Obstacle obstacle) {
        obstacles.remove(obstacle);
        if (selectedObstacle == obstacle) {
            selectedObstacle = null;
        }
        invalidate();
    }

    public void clearObstacles() {
        obstacles.clear();
        selectedObstacle = null;
        Obstacle.resetIdCounter();
        invalidate();
    }

    public List<Obstacle> getObstacles() {
        return new ArrayList<>(obstacles);
    }

    public Obstacle getSelectedObstacle() {
        return selectedObstacle;
    }

    public void setSelectedObstacle(Obstacle obstacle) {
        selectedObstacle = obstacle;
        invalidate();
    }

    public void clearSelection() {
        selectedObstacle = null;
        invalidate();
    }

    public void updateObstacle(Obstacle obstacle) {
        invalidate();
    }

    public int getGridSize() {
        return GRID_SIZE;
    }

    // Robot methods

    public void spawnRobot() {
        robot = new Robot();
        invalidate();
    }

    public void removeRobot() {
        robot = null;
        invalidate();
    }

    public Robot getRobot() {
        return robot;
    }

    public void setRobot(Robot robot) {
        this.robot = robot;
        invalidate();
    }

    public void updateRobotPosition(int x, int y, Robot.Direction direction) {
        if (robot == null) {
            robot = new Robot(x, y, direction);
        } else {
            robot.setGridX(x);
            robot.setGridY(y);
            robot.setFacing(direction);
        }
        invalidate();
    }

    public boolean hasRobot() {
        return robot != null;
    }

    // Lock methods

    public void setDragLocked(boolean locked) {
        this.isDragLocked = locked;
    }

    public boolean isDragLocked() {
        return isDragLocked;
    }
}
