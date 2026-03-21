package com.example.mdp_14;

import android.animation.Animator;
import android.animation.AnimatorListenerAdapter;
import android.animation.ValueAnimator;
import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Path;
import android.graphics.Rect;
import android.graphics.RectF;
import android.util.AttributeSet;
import android.view.GestureDetector;
import android.view.MotionEvent;
import android.view.View;
import android.view.animation.DecelerateInterpolator;

import androidx.annotation.ColorRes;
import androidx.core.content.ContextCompat;

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
    private Bitmap robotBitmap;
    private Paint bitmapPaint;
    private Paint obstaclePaint;
    private Paint obstacleDeletePaint;  // For drag-to-delete visual feedback
    private Paint targetIndicatorPaint;
    private Paint targetTextPaint;
    private Paint gridLabelPaint;
    private Paint selectedPaint;
    private Paint robotPaint;
    private Paint robotDirectionPaint;
    private Paint tooltipBgPaint;
    private Paint tooltipTextPaint;

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

    // Animated
    private float animatedRobotX = 0;
    private float animatedRobotY = 0;
    private float targetRobotX = 0;
    private float targetRobotY = 0;
    private boolean isAnimating = false;
    private ValueAnimator robotAnimator;

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
        gridPaint.setColor(ContextCompat.getColor(getContext(), R.color.grid_paint));
        gridPaint.setStrokeWidth(1f);
        gridPaint.setStyle(Paint.Style.STROKE);

        // Load the PNG from drawable
        robotBitmap = BitmapFactory.decodeResource(getResources(), R.drawable.robot_car);
        bitmapPaint = new Paint();
        bitmapPaint.setAntiAlias(true);
        bitmapPaint.setFilterBitmap(true); // Smooth scaling

        // Obstacle fill paint
        obstaclePaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        obstaclePaint.setColor(ContextCompat.getColor(getContext(), R.color.ink));
        obstaclePaint.setStyle(Paint.Style.FILL);

        // Obstacle delete preview paint (red, semi-transparent)
        obstacleDeletePaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        obstacleDeletePaint.setColor(ContextCompat.getColor(getContext(), R.color.obstacle_paint));
        obstacleDeletePaint.setStyle(Paint.Style.FILL);

        // Target indicator paint
        targetIndicatorPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        targetIndicatorPaint.setColor(Color.RED);
        targetIndicatorPaint.setStyle(Paint.Style.FILL);

        // Target text paint
        targetTextPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        targetTextPaint.setColor(ContextCompat.getColor(getContext(), R.color.bg_card));
        targetTextPaint.setTextAlign(Paint.Align.CENTER);

        // Grid label paint
        gridLabelPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        gridLabelPaint.setColor(ContextCompat.getColor(getContext(), R.color.ink_soft));
        gridLabelPaint.setTextSize(24f);
        gridLabelPaint.setTextAlign(Paint.Align.CENTER);

        // Selected highlight
        selectedPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        selectedPaint.setColor(ContextCompat.getColor(getContext(), R.color.sky));
        selectedPaint.setStrokeWidth(4f);
        selectedPaint.setStyle(Paint.Style.STROKE);

        // Tooltip paint
        tooltipBgPaint = new Paint();
        tooltipBgPaint.setColor(ContextCompat.getColor(getContext(), R.color.ink));
        tooltipBgPaint.setStyle(Paint.Style.FILL);
        tooltipBgPaint.setAntiAlias(true);

        tooltipTextPaint = new Paint();
        tooltipTextPaint.setColor(ContextCompat.getColor(getContext(), R.color.bg_card));
        tooltipTextPaint.setTextSize(18);
        tooltipTextPaint.setTextAlign(Paint.Align.CENTER);
        tooltipTextPaint.setAntiAlias(true);

        // Robot body paint
        robotPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        robotPaint.setColor(ContextCompat.getColor(getContext(), R.color.mint)); // Green
        robotPaint.setStyle(Paint.Style.FILL);

        // Robot direction indicator paint
        robotDirectionPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        robotDirectionPaint.setColor(ContextCompat.getColor(getContext(), R.color.robot_direction_paint)); // Dark green
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
        // Read padding set from XML or code
        float padTop    = getPaddingTop();
        float padBottom = getPaddingBottom();

        int width = getWidth();
        int height = getHeight();

        float labelMarginLeft = 30f;   // For row labels on left
        float labelMarginBottom = 30f; // For column labels at bottom

        float availableWidth = width - labelMarginLeft;
        float availableHeight = height - padTop  - padBottom - labelMarginBottom;

        cellSize = Math.min(availableWidth / GRID_SIZE, availableHeight / GRID_SIZE);

        float gridWidth = cellSize * GRID_SIZE;
        float gridHeight = cellSize * GRID_SIZE;
        offsetX = labelMarginLeft + (availableWidth - gridWidth) / 2;
        offsetY = padTop + (availableHeight - gridHeight) / 2;

        targetTextPaint.setTextSize(cellSize * 0.5f);
        gridLabelPaint.setTextSize(cellSize * 0.4f);
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);

        // Draw inset map background with rounded corners
        Paint mapBgPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        mapBgPaint.setColor(ContextCompat.getColor(getContext(), R.color.bg_inset));
        mapBgPaint.setStyle(Paint.Style.FILL);

        RectF mapRect = new RectF(
                offsetX,
                offsetY,
                offsetX + GRID_SIZE * cellSize,
                offsetY + GRID_SIZE * cellSize
        );
        float cornerRadius = 12f; // dp — adjust to taste
        canvas.drawRoundRect(mapRect, cornerRadius, cornerRadius, mapBgPaint);

//        canvas.drawColor(Color.WHITE);
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
            // If target has been recognized, show only the recognized target ID (no obstacle ID)
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

        if (!isOutsideGrid) {
            drawCoordinateTooltip(canvas, obstacle, centerX, top);
        }
    }

    private void drawCoordinateTooltip(Canvas canvas, Obstacle obstacle, float centerX, float obstacleTop) {
        String coordText = "(" + obstacle.getGridX() + "," + obstacle.getGridY() + ")";

        float textWidth = tooltipTextPaint.measureText(coordText);
        float paddingVertical = 20;
        float tooltipWidth = textWidth + cellSize * 0.8f;
        float tooltipHeight = tooltipTextPaint.getTextSize() + (paddingVertical * 2);

        float tooltipX = centerX - tooltipWidth / 2;
        float tooltipY = obstacleTop - tooltipHeight - cellSize * 0.3f;

        RectF tooltipRect = new RectF(tooltipX, tooltipY, tooltipX + tooltipWidth, tooltipY + tooltipHeight);
        canvas.drawRoundRect(tooltipRect, cellSize * 0.2f, cellSize * 0.2f, tooltipBgPaint);

        // Draw text at vertical center of tooltip
        float textCenterY = tooltipY + tooltipHeight / 2;
        Paint.FontMetrics fm = tooltipTextPaint.getFontMetrics();
        float textY = textCenterY - (fm.descent + fm.ascent) / 2;

        canvas.drawText(coordText, centerX, textY, tooltipTextPaint);
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

    private Bitmap createRobotBitmap(int sizePx) {
        Bitmap bmp = Bitmap.createBitmap(sizePx, sizePx, Bitmap.Config.ARGB_8888);
        Canvas c = new Canvas(bmp);

        float cx = sizePx / 2f;
        float cy = sizePx / 2f;
        float w  = sizePx * 0.82f;   // slightly smaller front by making overall narrower
        float h  = sizePx * 0.88f;

        float T  = cy - h * 0.44f;
        float B  = cy + h * 0.44f;
        // Front is now smaller — reduce hw slightly
        float hw = w * 0.38f;        // was 0.45f — makes front narrower

        int carColor  = ContextCompat.getColor(getContext(), R.color.sky);
        int darkColor = ContextCompat.getColor(getContext(), R.color.ink);

        Paint bodyPaint  = new Paint(Paint.ANTI_ALIAS_FLAG); bodyPaint.setColor(carColor); bodyPaint.setStyle(Paint.Style.FILL);
        Paint darkPaint  = new Paint(Paint.ANTI_ALIAS_FLAG); darkPaint.setColor(darkColor); darkPaint.setStyle(Paint.Style.FILL);
        Paint glassPaint = new Paint(Paint.ANTI_ALIAS_FLAG); glassPaint.setColor(Color.parseColor("#A0C8FF")); glassPaint.setStyle(Paint.Style.FILL);
        Paint tyrePaint  = new Paint(Paint.ANTI_ALIAS_FLAG); tyrePaint.setColor(Color.parseColor("#111111")); tyrePaint.setStyle(Paint.Style.FILL);
        Paint rimPaint   = new Paint(Paint.ANTI_ALIAS_FLAG); rimPaint.setColor(Color.parseColor("#CCCCCC")); rimPaint.setStyle(Paint.Style.FILL);
        Paint whitePaint = new Paint(Paint.ANTI_ALIAS_FLAG); whitePaint.setColor(Color.WHITE); whitePaint.setStyle(Paint.Style.FILL);
        Paint glowOut    = new Paint(Paint.ANTI_ALIAS_FLAG); glowOut.setColor(Color.parseColor("#2200FFB2")); glowOut.setStyle(Paint.Style.FILL);
        Paint glowMid    = new Paint(Paint.ANTI_ALIAS_FLAG); glowMid.setColor(Color.parseColor("#6600FFB2")); glowMid.setStyle(Paint.Style.FILL);
        Paint glowCore   = new Paint(Paint.ANTI_ALIAS_FLAG); glowCore.setColor(Color.parseColor("#00FFB2")); glowCore.setStyle(Paint.Style.FILL);

        // ── Main body ──────────────────────────────────────
        c.drawRoundRect(new RectF(cx - hw, T, cx + hw, B), w * 0.12f, w * 0.12f, bodyPaint);

        // ── Front bumper ───────────────────────────────────
        c.drawRoundRect(new RectF(cx - hw + 2, T - 4, cx + hw - 2, T + 6), 3, 3, darkPaint);
        // Bumper vents
        c.drawRoundRect(new RectF(cx - hw + 5, T - 2, cx - 4, T + 4), 1, 1, tyrePaint);
        c.drawRoundRect(new RectF(cx + 4, T - 2, cx + hw - 5, T + 4), 1, 1, tyrePaint);

        // ── Hood scoop ─────────────────────────────────────
        c.drawRoundRect(new RectF(cx - w*0.16f, T + h*0.02f, cx + w*0.16f, T + h*0.11f), 2, 2, darkPaint);

        // ── Roof / cabin ───────────────────────────────────
        c.drawRoundRect(new RectF(cx - hw + 7, T + h*0.14f, cx + hw - 7, T + h*0.56f), 4, 4, darkPaint);
        // Windscreen
        c.drawRoundRect(new RectF(cx - hw + 10, T + h*0.16f, cx + hw - 10, T + h*0.28f), 3, 3, glassPaint);
        // Rear cabin window
        c.drawRoundRect(new RectF(cx - hw + 10, T + h*0.42f, cx + hw - 10, T + h*0.53f), 3, 3, glassPaint);

        // ── Rear bumper ────────────────────────────────────
        c.drawRoundRect(new RectF(cx - hw + 2, B - 5, cx + hw - 2, B + 4), 3, 3, darkPaint);

        // ── Spoiler ────────────────────────────────────────
        c.drawRect(new RectF(cx - hw + 4, B + 2, cx + hw - 4, B + 7), darkPaint);
        c.drawRect(new RectF(cx - hw + 4, B - 3, cx - hw + 10, B + 8), darkPaint);
        c.drawRect(new RectF(cx + hw - 10, B - 3, cx + hw - 4, B + 8), darkPaint);

        // ── Racing stripes ─────────────────────────────────
        float sw = w * 0.075f;
        c.drawRect(new RectF(cx - sw*1.1f, T, cx - sw*0.1f, B), whitePaint);
        c.drawRect(new RectF(cx + sw*0.1f, T, cx + sw*1.1f, B), whitePaint);

        // ── 4 tyres ────────────────────────────────────────
        float fTW = w*0.17f, fTH = h*0.17f;
        float rTW = w*0.21f, rTH = h*0.23f;
        drawWheelOnCanvas(c, cx - hw - fTW*0.5f, T + fTH*0.9f, fTW, fTH, tyrePaint, rimPaint);
        drawWheelOnCanvas(c, cx + hw + fTW*0.5f, T + fTH*0.9f, fTW, fTH, tyrePaint, rimPaint);
        drawWheelOnCanvas(c, cx - hw - rTW*0.5f, B - rTH*0.9f, rTW, rTH, tyrePaint, rimPaint);
        drawWheelOnCanvas(c, cx + hw + rTW*0.5f, B - rTH*0.9f, rTW, rTH, tyrePaint, rimPaint);

        // ── Glow dot ───────────────────────────────────────
        float dotY = T + h*0.04f;
        float cr   = w*0.048f;
        c.drawCircle(cx, dotY, cr*3.2f, glowOut);
        c.drawCircle(cx, dotY, cr*1.9f, glowMid);
        c.drawCircle(cx, dotY, cr,      glowCore);

        return bmp;
    }

    private void drawWheelOnCanvas(Canvas c, float cx, float cy,
                                   float w, float h,
                                   Paint tyrePaint, Paint rimPaint) {
        Paint tread = new Paint(Paint.ANTI_ALIAS_FLAG);
        tread.setColor(Color.parseColor("#222222"));
        tread.setStyle(Paint.Style.STROKE);
        tread.setStrokeWidth(0.8f);

        Paint hub = new Paint(Paint.ANTI_ALIAS_FLAG);
        hub.setColor(Color.parseColor("#666666"));
        hub.setStyle(Paint.Style.FILL);

        c.drawRoundRect(new RectF(cx-w/2, cy-h/2, cx+w/2, cy+h/2), 4, 4, tyrePaint);
        for (int i = -1; i <= 1; i++)
            c.drawLine(cx-w/2+2, cy+i*h*0.28f, cx+w/2-2, cy+i*h*0.28f, tread);
        c.drawCircle(cx, cy, w*0.30f, rimPaint);
        c.drawCircle(cx, cy, w*0.13f, hub);
    }

    private void drawRobot(Canvas canvas) {
        if (robot == null) return;

        float drawX = isAnimating ? animatedRobotX : robot.getGridX();
        float drawY = isAnimating ? animatedRobotY : robot.getGridY();

        float left = offsetX + drawX * cellSize;
        float top  = offsetY + (GRID_SIZE - drawY - Robot.SIZE) * cellSize;
        float size = Robot.SIZE * cellSize;

        // Regenerate bitmap if size changed
        int sizePx = (int) size;
        if (robotBitmap == null || robotBitmap.getWidth() != sizePx) {
            robotBitmap = createRobotBitmap(sizePx);
        }

        canvas.save();
        float centerX  = left + size / 2f;
        float centerY  = top  + size / 2f;
        float rotation = getRotationAngle(robot.getFacing());
        canvas.rotate(rotation, centerX, centerY);

        canvas.drawBitmap(robotBitmap, left, top, null);

        canvas.restore();
    }

    // Using image
//    private void drawRobot(Canvas canvas) {
//        if (robot == null || robotBitmap == null) return;
//
//        float left = offsetX + robot.getGridX() * cellSize;
//        // Flip Y: gridY=0 at bottom
//        float top = offsetY + (GRID_SIZE - robot.getGridY() - Robot.SIZE) * cellSize;
//        float size = Robot.SIZE * cellSize;
//
//        // Save canvas state for rotation
//        canvas.save();
//
//        // Calculate center point for rotation
//        float centerX = left + size / 2;
//        float centerY = top + size / 2;
//
//        // Rotate based on direction
//        float rotation = getRotationAngle(robot.getFacing());
//        canvas.rotate(rotation, centerX, centerY);
//
//        // Draw the bitmap (with small padding like your original rect)
//        Rect srcRect = new Rect(0, 0, robotBitmap.getWidth(), robotBitmap.getHeight());
//        RectF destRect = new RectF(left + 3, top + 3, left + size - 3, top + size - 3);
//        canvas.drawBitmap(robotBitmap, srcRect, destRect, bitmapPaint);
//
//        // Restore canvas state
//        canvas.restore();
//    }

    private float getRotationAngle(Robot.Direction direction) {
        switch (direction) {
            case NORTH: return 0f;
            case EAST: return 90f;
            case SOUTH: return 180f;
            case WEST: return 270f;
            default: return 0f;
        }
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
        }

        // Store old position for animation
        float oldX = robot.getGridX();
        float oldY = robot.getGridY();

        robot.setGridX(x);
        robot.setGridY(y);
        robot.setFacing(direction);

        animateRobotMovement(oldX, oldY, x, y);
//        invalidate();
    }

    private void animateRobotMovement(float fromX, float fromY, float toX, float toY) {
        // Cancel any existing animation
        if (robotAnimator != null && robotAnimator.isRunning()) {
            robotAnimator.cancel();
        }

        animatedRobotX = fromX;
        animatedRobotY = fromY;
        targetRobotX = toX;
        targetRobotY = toY;

        robotAnimator = ValueAnimator.ofFloat(0f, 1f);
        robotAnimator.setDuration(500); // 500ms animation
        robotAnimator.setInterpolator(new DecelerateInterpolator());

        robotAnimator.addUpdateListener(animation -> {
            float progress = (float) animation.getAnimatedValue();
            animatedRobotX = fromX + (toX - fromX) * progress;
            animatedRobotY = fromY + (toY - fromY) * progress;
            isAnimating = true;
            invalidate(); // Redraw
        });

        robotAnimator.addListener(new AnimatorListenerAdapter() {
            @Override
            public void onAnimationEnd(Animator animation) {
                isAnimating = false;
                animatedRobotX = toX;
                animatedRobotY = toY;
                invalidate();
            }
        });

        robotAnimator.start();
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
