#include "main.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include "oled.h"
#include <stdarg.h>

/* ---- Machine-friendly messages over USART3 ---- */
static inline void ACK(const char *fmt, ...) {
  va_list ap; va_start(ap, fmt);
  printf("ACK:"); vprintf(fmt, ap); printf("\r\n");
  va_end(ap);
}
static inline void ERR(const char *fmt, ...) {
  va_list ap; va_start(ap, fmt);
  printf("ERR:"); vprintf(fmt, ap); printf("\r\n");
  va_end(ap);
}
static inline void DONE(const char *fmt, ...) {
  va_list ap; va_start(ap, fmt);
  printf("DONE:"); vprintf(fmt, ap); printf("\r\n");
  va_end(ap);
}

/* ---------------- HAL handles (as in your project) ---------------- */
ADC_HandleTypeDef hadc1;
I2C_HandleTypeDef hi2c2;
TIM_HandleTypeDef htim1;
TIM_HandleTypeDef htim2;
TIM_HandleTypeDef htim3;
TIM_HandleTypeDef htim4;
TIM_HandleTypeDef htim5;
TIM_HandleTypeDef htim8;
TIM_HandleTypeDef htim11;
TIM_HandleTypeDef htim12;
UART_HandleTypeDef huart2;
UART_HandleTypeDef huart3;
#define CMD_BUF_LEN 64
char cmd_buf[CMD_BUF_LEN];
int cmd_index = 0;
volatile uint32_t no_of_tick = 0;
volatile float position = 0;
volatile float speed = 0;
/* -------------------- Prototypes from CubeMX ---------------------- */
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_TIM8_Init(void);
static void MX_TIM2_Init(void);
static void MX_USART2_UART_Init(void);
static void MX_TIM1_Init(void);
static void MX_USART3_UART_Init(void);
static void MX_I2C2_Init(void);
static void MX_TIM5_Init(void);
static void MX_TIM4_Init(void);
static void MX_TIM3_Init(void);
static void MX_TIM11_Init(void);
static void MX_TIM12_Init(void);
static void MX_ADC1_Init(void);

/* ---------------------------- Globals ----------------------------- */
/* IMU globals (matching your project) */
volatile float ax_g, ay_g, az_g, gx_dps, gy_dps, gz_dps;

/* Global Angle helpers */
static float g_straight_dev_deg = 0.0f;
static float g_turn_residual_deg = 0.0f;
#define TURN_RESIDUAL_MIN_DEG (-10.0f)
#define TURN_RESIDUAL_MAX_DEG (10.0f)

/* Motor/Motion control */
static int PWM_TRIM = 650;       // negative slows the left side to fix veer-right
static const int16_t pwmMax = (7200 - 200);
static const int16_t pwmMin = 800;

// Encoder calibration (3rd pass):
static float COUNTS_PER_CM_L = 67.65 * 1.0f;
static float COUNTS_PER_CM_R = 65.49 * 1.0f;



/* Debug / UI */
char buf[128];

/* --------------------- IMU (ICM-20948) bits ----------------------- */
#define WHO_AM_I      0x00
#define WHO_AM_I_VAL  0xEA
#define REG_BANK_SEL  0x7F
#define ICM_ADDR_68   (0x68 << 1)     // AD0 = 0
#define ICM_ADDR_69   (0x69 << 1)     // AD0 = 1
static uint16_t ICM_ADDR = ICM_ADDR_69;

/* ---------------------- Forward declarations ---------------------- */

static inline void Servo_WriteUS(uint16_t us);
static uint16_t Steering_ToUS(int16_t steer_angle);
static void MotorDrive_enable(void);
static void Motor_stop(void);
static void Motor_forward(int pwmVal);
static void Motor_reverse(int pwmVal);
void Display_UART_Command(const char *cmd)
{
    char display_buf[32];
    snprintf(display_buf, sizeof(display_buf), "CMD: %s", cmd);
    OLED_Clear();
    OLED_ShowString(0, 20, (const uint8_t *)display_buf);
    OLED_Refresh_Gram();
}
/* UART printf redirect */
int _write(int file, char *ptr, int len);


/* Turn helpers */
static void Gyro_ResetHeading(void);
static void Gyro_UpdateFromIMU(float dt_s);
static float Gyro_GetHeadingDeg(void);
static float Drive_Turn_Angle(float target_deg, int steer_deg, int base_pwm);
static float Drive_Turn_AngleBW(float target_deg, int steer_deg, int base_pwm);

/* Straight helpers */
static inline void reset_encoders(void);
static inline int32_t left_ticks(void);
static inline int32_t right_ticks(void);
static float cm_travelled(void);
static int pwm_for_distance(float cm_left, int base_pwm);
static void Drive_Straight_ToCM(float target_cm, int base_pwm);
static uint32_t HCSR04_Read(void);   // <-- ADD THIS
static uint8_t EmergencyStop_Usonic(void);

/* --------------------- Helpers / small drivers -------------------- */
static inline void Servo_WriteUS(uint16_t us) {
  if (us < 500)  us = 500;
  if (us > 2500) us = 2500;
  __HAL_TIM_SET_COMPARE(&htim12, TIM_CHANNEL_2, us);
}
// ===== SERVO CONFIG =====

// Adjust this so wheels are perfectly straight at steer_angle = 0
static uint16_t SERVO_US_CENTER = 1500;
static uint16_t SERVO_US_LEFT_MAX = 1900;
static uint16_t SERVO_US_RIGHT_MAX = 1900;

// These MUST be tuned so wheels never hit the chassis
#define SERVO_US_MIN   1100
#define SERVO_US_MAX   2200

// Maximum logical steering angle allowed by your chassis
#define STEER_ANGLE_LIMIT  40     // degrees
#define STEER_ANGLE_LIMIT_RIGHT 45

// Servo scaling: µs per degree
#define SERVO_US_PER_DEG  ((2400 - 500) / 90.0f)   // ≈21.1 µs/deg

static uint16_t Steering_ToUS_original(int16_t steer_angle) { //new update: calculation works for right turn
  if (steer_angle < -45) steer_angle = -45;
  if (steer_angle >  45) steer_angle =  65;
  int32_t us = SERVO_US_CENTER + (int32_t)steer_angle * ((2400 - 500) / 90);//this speific calculation works, add to new steering function
  __HAL_TIM_SET_COMPARE(&htim12, TIM_CHANNEL_2, (uint16_t)us);
  return (uint16_t)us;
}

/*static uint16_t Steering_ToUS(int16_t steer_angle)// working, just right servo not turning enuf
{
    // 1️⃣ Clamp steering angle to safe mechanical limit
	if (steer_angle < -STEER_ANGLE_LIMIT) {
	    steer_angle = -STEER_ANGLE_LIMIT;
	}
	if (steer_angle > STEER_ANGLE_LIMIT) {
	    steer_angle = STEER_ANGLE_LIMIT;
	}

    // 2️⃣ Convert angle to pulse width
    int32_t us = SERVO_US_CENTER +
                 (int32_t)(steer_angle * SERVO_US_PER_DEG);

    // 3️⃣ Hard safety clamp on pulse width
    if (us < SERVO_US_MIN)
        us = SERVO_US_MIN;
    if (us > SERVO_US_MAX)
        us = SERVO_US_MAX;

    // 4️⃣ Apply to timer
    __HAL_TIM_SET_COMPARE(&htim12, TIM_CHANNEL_2, (uint16_t)us);

    return (uint16_t)us;
}*/

static uint16_t Steering_ToUS(int16_t steer_angle)//tested, works for left turn
{
    // 1️⃣ Clamp the steering angle to limits
    if (steer_angle < -STEER_ANGLE_LIMIT)
        steer_angle = -STEER_ANGLE_LIMIT;

    int32_t us;

    // 2️⃣ Compute PWM based on side-specific range
    if (steer_angle < 0)
    {
        // Left turn
        us = SERVO_US_CENTER +
             (int32_t)(steer_angle * (SERVO_US_CENTER - SERVO_US_LEFT_MAX) / -STEER_ANGLE_LIMIT);//calculation is correct for left turn
    }
    else if (steer_angle > 0)
    {
        // Right turn
        //us = SERVO_US_CENTER +
         //(int32_t)(steer_angle * (SERVO_US_RIGHT_MAX - SERVO_US_CENTER) / STEER_ANGLE_LIMIT);
        //replace above formula with this and test
         us = SERVO_US_CENTER + (int32_t)steer_angle * ((2400 - 500) / 90);//this speific calculation works, add to new steering function
    }
    else
    {
        // Center
        us = SERVO_US_CENTER;
    }

    // 3️⃣ Hard safety clamp on pulse width
    if (us < SERVO_US_MIN)
        us = SERVO_US_MIN;
    if (us > SERVO_US_MAX)
        us = SERVO_US_MAX;

    // 4️⃣ Apply to timer
    __HAL_TIM_SET_COMPARE(&htim12, TIM_CHANNEL_2, (uint16_t)us);

    return (uint16_t)us;
}


static uint16_t Steering_ToUS_Fixed(int16_t steer_angle)
{
    // Clamp
    if(steer_angle < -STEER_ANGLE_LIMIT) steer_angle = -STEER_ANGLE_LIMIT;
    if(steer_angle > STEER_ANGLE_LIMIT_RIGHT) steer_angle = STEER_ANGLE_LIMIT_RIGHT;

    int32_t us;

    if(steer_angle < 0) { // left turn
        us = SERVO_US_CENTER + (int32_t)((float)steer_angle / -STEER_ANGLE_LIMIT * (SERVO_US_CENTER - SERVO_US_LEFT_MAX));
    } else if(steer_angle > 0) { // right turn
        us = SERVO_US_CENTER + (int32_t)((float)steer_angle / STEER_ANGLE_LIMIT_RIGHT * (SERVO_US_RIGHT_MAX - SERVO_US_CENTER));
    } else {
        us = SERVO_US_CENTER;
    }

    // Clamp just in case
    if(us < SERVO_US_MIN) us = SERVO_US_MIN;
    if(us > SERVO_US_MAX) us = SERVO_US_MAX;

    __HAL_TIM_SET_COMPARE(&htim12, TIM_CHANNEL_2, (uint16_t)us);

    return (uint16_t)us;

  }


/* ----------------------- IMU I2C helpers -------------------------- */
static void ICM20948_SelectBank(uint8_t bank) {
  uint8_t d[2] = {REG_BANK_SEL, (uint8_t)(bank << 4)};
  HAL_I2C_Master_Transmit(&hi2c2, ICM_ADDR, d, 2, HAL_MAX_DELAY);
}
static void ICM20948_WriteReg(uint8_t bank, uint8_t reg, uint8_t val) {
  ICM20948_SelectBank(bank);
  uint8_t d[2] = {reg, val};
  HAL_I2C_Master_Transmit(&hi2c2, ICM_ADDR, d, 2, HAL_MAX_DELAY);
}
static void ICM20948_ReadRegs(uint8_t bank, uint8_t reg, uint8_t *data, uint8_t len) {
  ICM20948_SelectBank(bank);
  HAL_I2C_Master_Transmit(&hi2c2, ICM_ADDR, &reg, 1, HAL_MAX_DELAY);
  HAL_I2C_Master_Receive(&hi2c2, ICM_ADDR, data, len, HAL_MAX_DELAY);
}
static HAL_StatusTypeDef icm_read_raw(uint16_t addr, uint8_t reg, uint8_t *data, uint8_t len) {
  if (HAL_I2C_Master_Transmit(&hi2c2, addr, &reg, 1, 100) != HAL_OK) return HAL_ERROR;
  return HAL_I2C_Master_Receive(&hi2c2, addr, data, len, 100);
}
static int ICM20948_Detect(void) {
  uint8_t who = 0;
  if (icm_read_raw(ICM_ADDR_68, WHO_AM_I, &who, 1) == HAL_OK && who == WHO_AM_I_VAL) { ICM_ADDR = ICM_ADDR_68; return 0; }
  if (icm_read_raw(ICM_ADDR_69, WHO_AM_I, &who, 1) == HAL_OK && who == WHO_AM_I_VAL) { ICM_ADDR = ICM_ADDR_69; return 0; }
  return -1;
}
static int ICM20948_Init(void) {
  uint8_t whoami;
  ICM20948_ReadRegs(0, WHO_AM_I, &whoami, 1);
  if (whoami != WHO_AM_I_VAL) return -1;
  ICM20948_WriteReg(0, 0x06, 0x80); HAL_Delay(100);
  ICM20948_WriteReg(0, 0x06, 0x01);
  ICM20948_WriteReg(0, 0x07, 0x00);
  ICM20948_WriteReg(0, 0x05, 0x00);
  ICM20948_WriteReg(2, 0x14, 0x00);
  ICM20948_WriteReg(2, 0x01, 0x00);
  return 0;
}
static void ICM20948_ReadRaw(int16_t *ax, int16_t *ay, int16_t *az,
                             int16_t *gx, int16_t *gy, int16_t *gz) {
  uint8_t d[12];
  ICM20948_ReadRegs(0, 0x2D, d, 12);
  *ax = (d[0] << 8) | d[1]; *ay = (d[2] << 8) | d[3]; *az = (d[4] << 8) | d[5];
  *gx = (d[6] << 8) | d[7]; *gy = (d[8] << 8) | d[9]; *gz = (d[10] << 8) | d[11];
}

/* ------------------- Motor drive / direction ---------------------- */
static void MotorDrive_enable(void) {
  HAL_TIM_PWM_Start(&htim4, TIM_CHANNEL_3); // Motor A (left)
  HAL_TIM_PWM_Start(&htim4, TIM_CHANNEL_4);
  HAL_TIM_PWM_Start(&htim1, TIM_CHANNEL_3); // Motor D (right)
  HAL_TIM_PWM_Start(&htim1, TIM_CHANNEL_4);
}
static void Motor_stop(void) {
  __HAL_TIM_SetCompare(&htim4, TIM_CHANNEL_3, 0);
  __HAL_TIM_SetCompare(&htim4, TIM_CHANNEL_4, 0);
  __HAL_TIM_SetCompare(&htim1, TIM_CHANNEL_3, 0);
  __HAL_TIM_SetCompare(&htim1, TIM_CHANNEL_4, 0);
}

/* ------------------- Heading-hold Motor_forward ------------------- */
#define MOTOR_TRIM_LEFT   1.097f
#define MOTOR_TRIM_RIGHT  0.867f


static void Motor_forward(int pwmVal) { //not using
  static uint32_t last_time = 0;
  static float heading = 0.0f, target_heading = 0.0f;
  static uint8_t initialized = 0;
  static float integral = 0.0f, last_error = 0.0f;
  static float gz_filtered = 0.0f;

  uint32_t now = HAL_GetTick();
  float dt = (now - last_time) / 1000.0f;
  if (dt <= 0) dt = 0.001f;
  last_time = now;

  if (!initialized) { target_heading = heading; initialized = 1; }

  gz_filtered = 0.9f * gz_filtered + 0.1f * gz_dps;
  heading += gz_filtered * dt;

  float err = target_heading - heading;
  integral += err * dt;
  float deriv = (err - last_error) / dt;
  last_error = err;

  const float Kp = 40.0f, Ki = 4.0f, Kd = 3.5f;
  int correction = (int)(Kp * err + Ki * integral + Kd * deriv);
  if (correction > 2000)  correction = 2000;
  if (correction < -2000) correction = -2000;

  int left_pwm  = pwmVal + PWM_TRIM  + correction;
  int right_pwm = pwmVal - PWM_TRIM  - correction;

  if (left_pwm > pwmMax)  left_pwm = pwmMax;
  if (left_pwm < pwmMin)  left_pwm = pwmMin;
  if (right_pwm > pwmMax) right_pwm = pwmMax;
  if (right_pwm < pwmMin) right_pwm = pwmMin;

  __HAL_TIM_SetCompare(&htim4, TIM_CHANNEL_3, left_pwm);
  __HAL_TIM_SetCompare(&htim4, TIM_CHANNEL_4, 0);

  __HAL_TIM_SetCompare(&htim1, TIM_CHANNEL_3, 0);
  __HAL_TIM_SetCompare(&htim1, TIM_CHANNEL_4, right_pwm);
}

static void Motor_forward_raw(int pwmVal) { //  not using
  //int left_pwm  = pwmVal + PWM_TRIM;
  //int right_pwm = pwmVal - PWM_TRIM;
  int left_pwm  = (int)(pwmVal * MOTOR_TRIM_LEFT);
  int right_pwm = (int)(pwmVal * MOTOR_TRIM_RIGHT);
  if (left_pwm  < pwmMin) left_pwm  = pwmMin;
  if (right_pwm < pwmMin) right_pwm = pwmMin;
  if (left_pwm  > pwmMax) left_pwm  = pwmMax;
  if (right_pwm > pwmMax) right_pwm = pwmMax;

  __HAL_TIM_SetCompare(&htim4, TIM_CHANNEL_3, left_pwm);
  __HAL_TIM_SetCompare(&htim4, TIM_CHANNEL_4, 0);
  __HAL_TIM_SetCompare(&htim1, TIM_CHANNEL_3, 0);
  __HAL_TIM_SetCompare(&htim1, TIM_CHANNEL_4, right_pwm);
}

static void Motor_forward_turn_working(int base_pwm, int steer_deg)//can turn left and right forwad only, not using
{

    int left_pwm  = base_pwm;
    int right_pwm = base_pwm;

    // Simple differential scaling
    float turn_factor = fabsf(steer_deg) / 30.0f; // assuming 30° max steer
    if (turn_factor > 1.0f) turn_factor = 1.0f;

    int delta = (int)(base_pwm * 0.3f * turn_factor);

    if (steer_deg > 0) {
        // turning right
        left_pwm  += delta;
        right_pwm -= delta;
    } else {
        // turning left
        left_pwm  -= delta;
        right_pwm += delta;
    }

    // Clamp
    if (left_pwm  < pwmMin) left_pwm  = pwmMin;
    if (right_pwm < pwmMin) right_pwm = pwmMin;
    if (left_pwm  > pwmMax) left_pwm  = pwmMax;
    if (right_pwm > pwmMax) right_pwm = pwmMax;

    __HAL_TIM_SetCompare(&htim4, TIM_CHANNEL_3, left_pwm);
    __HAL_TIM_SetCompare(&htim4, TIM_CHANNEL_4, 0);
    __HAL_TIM_SetCompare(&htim1, TIM_CHANNEL_3, 0);
    __HAL_TIM_SetCompare(&htim1, TIM_CHANNEL_4, right_pwm);
}

#define max_right 40
#define max_left 45
static void Motor_drive_turn(int base_pwm, int steer_deg, int direction)//works for backwards and forwards turning
{
    int left_pwm  = base_pwm;
    int right_pwm = base_pwm;

    float turn_factor = 0;



    if(steer_deg > 0){
    	turn_factor = fabsf(steer_deg) / max_right;
    }
    else if(steer_deg < 0){

    	turn_factor = fabsf(steer_deg) / max_left;
    }else{
    	turn_factor = 0;
    }

    if (turn_factor > 1.0f) turn_factor = 1.0f;

    int delta = (int)(base_pwm * 0.6f * turn_factor);

    int turn_sign = (steer_deg > 0) ? 1 : -1;

    // Invert differential if reversing
    if (direction < 0)
        turn_sign = -turn_sign;

    if (turn_sign > 0) {
        left_pwm  += delta;
        right_pwm -= delta;
    } else {
        left_pwm  -= delta;
        right_pwm += delta;
    }

    // APPLY TRIM AFTER DIFFERENTIAL
    if(direction > 0){
    	if(steer_deg > 0){// turning right
    		left_pwm  = (int)(left_pwm  *  MOTOR_TRIM_LEFT);//1.097
    		right_pwm = (int)(right_pwm * MOTOR_TRIM_RIGHT);//0.867
    	}else{// turning left
    		left_pwm  = (int)(left_pwm  *  MOTOR_TRIM_RIGHT);//0.867
    		right_pwm = (int)(right_pwm * MOTOR_TRIM_LEFT);//1.097
    	}

    }
    else{
    	if(steer_deg < 0){// turning right
    		left_pwm  = (int)(left_pwm  * 0.867);
    		right_pwm = (int)(right_pwm * 1.097);
    	}else{//turning left
    		left_pwm  = (int)(left_pwm  * 1.097);
    		right_pwm = (int)(right_pwm * 0.867);
    	}

    }
    //left_pwm  = (int)(left_pwm  *  MOTOR_TRIM_LEFT);
    //right_pwm = (int)(right_pwm * MOTOR_TRIM_RIGHT);

    //left_pwm += 0;
    //right_pwm -= 0;
    // Clamp
    if (left_pwm  < pwmMin) left_pwm  = pwmMin;
    if (right_pwm < pwmMin) right_pwm = pwmMin;
    if (left_pwm  > pwmMax) left_pwm  = pwmMax;
    if (right_pwm > pwmMax) right_pwm = pwmMax;

    if (direction > 0) {
        // forward
        __HAL_TIM_SetCompare(&htim4, TIM_CHANNEL_3, left_pwm);
        __HAL_TIM_SetCompare(&htim4, TIM_CHANNEL_4, 0);
        __HAL_TIM_SetCompare(&htim1, TIM_CHANNEL_3, 0);
        __HAL_TIM_SetCompare(&htim1, TIM_CHANNEL_4, right_pwm);
    } else {
        // reverse (swap direction pins)
        __HAL_TIM_SetCompare(&htim4, TIM_CHANNEL_3, 0);
        __HAL_TIM_SetCompare(&htim4, TIM_CHANNEL_4, left_pwm);
        __HAL_TIM_SetCompare(&htim1, TIM_CHANNEL_3, right_pwm);
        __HAL_TIM_SetCompare(&htim1, TIM_CHANNEL_4, 0);
    }
}


/* Optional reverse */
static void Motor_reverse(int pwmVal) {
  __HAL_TIM_SetCompare(&htim4, TIM_CHANNEL_3, 0);
  __HAL_TIM_SetCompare(&htim4, TIM_CHANNEL_4, pwmVal);
  __HAL_TIM_SetCompare(&htim1, TIM_CHANNEL_3, pwmVal);
  __HAL_TIM_SetCompare(&htim1, TIM_CHANNEL_4, 0);
}

/* ---------------------- Encoder utilities ------------------------- */
/* TIM2: PA15/PB3  (Left), TIM5: PA0/PA1 (Right) — confirmed in MSP. */
static uint16_t L0 = 0, R0 = 0;  // baselines captured at reset

static inline uint32_t tim_arr(TIM_TypeDef *t) {
  return (t->ARR ? t->ARR : 0xFFFFu);
}

// Forward ticks on LEFT wheel since reset (>=0) with wrap handling
static inline int32_t left_ticks(void) {
  uint32_t arrp1 = tim_arr(TIM2) + 1u;
  int32_t d = (int32_t)((uint32_t)TIM2->CNT - (uint32_t)L0);
  if (d < 0) d += (int32_t)arrp1;
  return d;
}

// Forward ticks on RIGHT wheel since reset (>=0); flip sense (R reversed)
static inline int32_t right_ticks(void) {
  uint32_t arrp1 = tim_arr(TIM5) + 1u;
  int32_t d = (int32_t)((int32_t)R0 - (int32_t)TIM5->CNT);
  if (d < 0) d += (int32_t)arrp1;
  return d;
}

static inline void reset_encoders(void) {
  __HAL_TIM_SET_COUNTER(&htim2, 0);
  __HAL_TIM_SET_COUNTER(&htim5, 0);
  HAL_Delay(2);                   // small settle
  L0 = (uint16_t)TIM2->CNT;      // capture baselines
  R0 = (uint16_t)TIM5->CNT;
}

// average distance using the calibrated scales
static float cm_travelled(void) {
  float cmL = (float)left_ticks()  / COUNTS_PER_CM_L;
  float cmR = (float)right_ticks() / COUNTS_PER_CM_R;
  return 0.5f * (cmL + cmR);
}

/* --- NEW: signed tick + cm helpers (reverse becomes negative) --- */
static inline int32_t left_ticks_signed(void) {
  return (int16_t)((int32_t)TIM2->CNT - (int32_t)L0);
}
static inline int32_t right_ticks_signed(void) {
  return (int16_t)((int32_t)R0 - (int32_t)TIM5->CNT); // keep flipped sense on R
}
static float cm_travelled_signed(void) {
  float cmL = (float)left_ticks_signed()  / COUNTS_PER_CM_L;
  float cmR = (float)right_ticks_signed() / COUNTS_PER_CM_R;
  return 0.5f * (cmL + cmR);  // +forward, -reverse
}

/* --- NEW: generic signed-distance mover (+cm fwd, -cm rev) --- */
static void Drive_Move_ByCM(float target_cm, int base_pwm) {
  reset_encoders();
  const float STOP_TOL_CM = fmaxf(2.0f, fabsf(target_cm) * 0.06f);
  const int dir = (target_cm >= 0.0f) ? +1 : -1;
  uint32_t lastPrint = 0;

  while (1) {
    float done_cm  = cm_travelled_signed();
    float left_cm  = target_cm - done_cm;
    if (fabsf(left_cm) <= STOP_TOL_CM) break;

    int pwm = pwm_for_distance(fabsf(left_cm), base_pwm);
    if (pwm < 260) pwm = 260;
    if (dir > 0) Motor_forward(pwm); else Motor_reverse(pwm);

    uint32_t now = HAL_GetTick();
    if (now - lastPrint >= 100) {
      lastPrint = now;
      printf("MOVE cm_done:%.1f cm_left:%.1f dir:%c pwm:%d\r\n",
             done_cm, left_cm, (dir>0)?'+':'-', pwm);
    }
    HAL_Delay(10);
  }
  Motor_stop();
  printf("[STOP MOVE] cm_done:%.1f target:%.1f\r\n",
         cm_travelled_signed(), target_cm);
}


/* -------------------- Speed ramp / braking ------------------------ */
static int pwm_for_distance(float cm_left, int base_pwm) {
  if (cm_left > 30.0f)      return base_pwm;
  else if (cm_left > 10.0f) return (int)(base_pwm * 0.60f);
  else if (cm_left > 3.0f)  return (int)(base_pwm * 0.35f);
  else                      return (int)(base_pwm * 0.25f);
}

/// -----------------------------
// CONFIGURATION
// -----------------------------
#define MOTOR_TRIM    640//need to modify for tuning
// tune until robot drives straight (forward/backward)
#define HEADING_KP    20.0f //PID values dont really matter
#define HEADING_KI    0.9f
#define HEADING_KD    2.5f

// -----------------------------
static inline float clampf(float v, float lo, float hi) {
  if (v < lo) return lo;
  if (v > hi) return hi;
  return v;
}
static inline int clampi(int v, int lo, int hi) {
  if (v < lo) return lo;
  if (v > hi) return hi;
  return v;
}

void Encoder_Calibration_Test(void)
{
    char buf[32];

    reset_encoders();
    Steering_ToUS(30);
    Steering_ToUS(0);
    HAL_Delay(500);
    set_left_motor(1800);
    set_right_motor(1000);
    uint32_t start = HAL_GetTick();

    while (HAL_GetTick() - start < 5000)
    {
        OLED_Clear();

        snprintf(buf, sizeof(buf), "ENCODER TEST");
        OLED_ShowString(0, 0, (uint8_t*)buf);

        snprintf(buf, sizeof(buf), "Running...");
        OLED_ShowString(0, 16, (uint8_t*)buf);

        snprintf(buf, sizeof(buf), "Time:%lu", (HAL_GetTick() - start)/1000);
        OLED_ShowString(0, 28, (uint8_t*)buf);

        OLED_Refresh_Gram();     // <<< THIS is why Drive_Turn works

        HAL_Delay(100);
    }

    Motor_stop();
    HAL_Delay(500);

    int32_t left_ticks  = left_ticks_signed();
    int32_t right_ticks = right_ticks_signed();

    OLED_Clear();

    snprintf(buf, sizeof(buf), "L Ticks:%ld", (long)left_ticks);
    OLED_ShowString(0, 0, (uint8_t*)buf);

    snprintf(buf, sizeof(buf), "R Ticks:%ld", (long)right_ticks);
    OLED_ShowString(0, 16, (uint8_t*)buf);

    OLED_ShowString(0, 28, (uint8_t*)"Measure CM");
    OLED_ShowString(0, 40, (uint8_t*)"Manual Calc");

    OLED_Refresh_Gram();         // <<< also needed here
}

void Drive_Straight_ToCM(float target_cm, int base_pwm)
{
    reset_encoders();
    Steering_ToUS(45);
    HAL_Delay(500);
    Steering_ToUS(1);
    HAL_Delay(500);

    // Let gyro settle then zero it
    Gyro_ResetHeading();
    HAL_Delay(200);
    Gyro_ResetHeading();

    const int dir = (target_cm >= 0.0f) ? +1 : -1;
    target_cm = fabsf(target_cm);

    const float STOP_TOL_CM = 0.0f;

    float integral = 0.0f;
    float last_error = 0.0f;

    uint32_t last_time = HAL_GetTick();
    uint32_t last_oled = last_time;

    char buf[32];

    while (1)
    {
    	if (dir > 0)
    	{
    	    uint32_t distance_cm = HCSR04_Read();
    	    if (distance_cm <= 20)
    	    {
    	        Motor_stop();
    	        HAL_GPIO_WritePin(GPIOA, GPIO_PIN_8, GPIO_PIN_SET);
    	        HAL_Delay(500);
    	        HAL_GPIO_WritePin(GPIOA, GPIO_PIN_8, GPIO_PIN_RESET);
    	        OLED_ShowString(0, 40, "Obstacle detected!");
    	        break;
    	    }
    	}
        // ---- timing ----
        uint32_t now = HAL_GetTick();
        float dt = (now - last_time) / 1000.0f;
        if (dt <= 0.0f) dt = 0.001f;
        last_time = now;

        // ---- update gyro ----
        Gyro_UpdateFromIMU(dt);
        float heading = Gyro_GetHeadingDeg();   // YOUR convention: +left, -right

        // ---- distance from encoders ----
        float left_cm  = left_ticks_signed()  / COUNTS_PER_CM_L;
        float right_cm = right_ticks_signed() / COUNTS_PER_CM_R;
        float avg_cm   = (left_cm + right_cm) * 0.5f;  // signed
        float done_cm  = fabsf(avg_cm);
        float remaining = target_cm - done_cm;

        if (remaining <= STOP_TOL_CM) break;

        // ---- speed profile ----
        int pwm = base_pwm;
        if      (remaining > 30.0f) pwm = base_pwm;
        else if (remaining > 10.0f) pwm = (int)(base_pwm * 0.60f);
        else if (remaining >  5.0f) pwm = (int)(base_pwm * 0.45f);
        else                        pwm = (int)(base_pwm * 0.35f);

        // keep above deadzone
        const int pwm_min = 280;
        if (pwm < pwm_min) pwm = pwm_min;

        // ---- heading PID (target heading = 0) ----
        // Since +heading means left drift, error should be +heading
        float error = heading;

        integral += error * dt;
        integral = clampf(integral, -25.0f, 25.0f);

        float derivative = (error - last_error) / dt;
        last_error = error;

        int corr = (int)(HEADING_KP * error + HEADING_KI * integral + HEADING_KD * derivative);

        // Limit correction relative to speed
        int corr_limit = (int)(0.20f * pwm);
        if (corr_limit < 60) corr_limit = 60;
        corr = clampi(corr, -corr_limit, corr_limit);

        // ---- motor mix: trim + gyro ----
        // MOTOR_TRIM should be small (e.g., 20–80). Positive trim boosts left, reduces right.

        int base = dir * pwm;
        int trim = dir * (MOTOR_TRIM * pwm) / base_pwm;
        //int trim = MOTOR_TRIM;
        // corr>0 (left drift) -> make RIGHT faster to steer right
        int left_cmd  = base + trim - corr;
        int right_cmd = base - trim + corr;

        left_cmd  = clampi(left_cmd,  -pwmMax, pwmMax);
        right_cmd = clampi(right_cmd, -pwmMax, pwmMax);

        set_left_motor(left_cmd);
        set_right_motor(right_cmd);

        // ---- OLED debug ----
        if (now - last_oled >= 100u)
        {
            last_oled = now;

            OLED_Clear();

            snprintf(buf, sizeof(buf), "Head:%.2f", heading);
            OLED_ShowString(0, 0, (uint8_t*)buf);

            snprintf(buf, sizeof(buf), "Rem:%.1f", remaining);
            OLED_ShowString(0, 16, (uint8_t*)buf);

            snprintf(buf, sizeof(buf), "L:%d R:%d", left_cmd, right_cmd);
            OLED_ShowString(0, 28, (uint8_t*)buf);

            snprintf(buf, sizeof(buf), "PWM:%d C:%d", pwm, corr);
            OLED_ShowString(0, 40, (uint8_t*)buf);

            OLED_Refresh_Gram();
        }

        HAL_Delay(10);
    }

    Motor_stop();
    HAL_Delay(80);

    float final_heading = Gyro_GetHeadingDeg();

    g_straight_dev_deg = final_heading;

    //if (g_straight_dev_deg > 15.0f)  g_straight_dev_deg = 15.0f;
    //if (g_straight_dev_deg < -15.0f) g_straight_dev_deg = -15.0f;

    printf("STRAIGHT_DEV stored=%.2f final_heading=%.2f\r\n",
           g_straight_dev_deg, final_heading);

    OLED_Clear();
    snprintf(buf, sizeof(buf), "DONE Head:%.2f", final_heading);
    OLED_ShowString(0, 0, (uint8_t*)buf);
    OLED_Refresh_Gram();
}
void Drive_Straight_ToCM_USONIC(float target_cm, int base_pwm) // main forward and backwards code
{
    // -----------------------------
    // 1. Initialization
    // -----------------------------
    reset_encoders();
    Steering_ToUS(0);   // center steering

    const float STOP_TOL_CM = 2.0f;
    const uint32_t OBSTACLE_STOP_CM = 20;   // <-- emergency stop distance (cm)

    const int dir = (target_cm >= 0.0f) ? +1 : -1;
    target_cm = fabsf(target_cm);

    float heading = 0.0f;
    float integral = 0.0f;
    float last_error = 0.0f;
    uint32_t last_time = HAL_GetTick();

    // -----------------------------
    // 2. Main Loop
    // -----------------------------
    while (1)
    {
        // ----- Emergency stop for obstacles -----
        uint32_t distance_cm = HCSR04_Read();
        if (distance_cm <= OBSTACLE_STOP_CM)
        {
            Motor_stop();
            OLED_ShowString(0, 40, "Obstacle detected!");
            break;
        }

        // ----- Time Step -----
        uint32_t now = HAL_GetTick();
        float dt = (now - last_time) / 1000.0f;
        if (dt <= 0) dt = 0.001f;
        last_time = now;

        // ----- Distance from Encoders -----
        float left_cm  = left_ticks_signed()  / COUNTS_PER_CM_L;
        float right_cm = right_ticks_signed() / COUNTS_PER_CM_R;

        float avg_cm = (fabsf(left_cm) + fabsf(right_cm)) * 0.5f;
        float remaining = target_cm - avg_cm;

        if (remaining <= STOP_TOL_CM)
            break;

        // -----------------------------
        // 3. Speed Profile (Soft Braking)
        // -----------------------------
        int pwm = base_pwm;

        if      (remaining > 30.0f) pwm = base_pwm;
        else if (remaining > 10.0f) pwm = (int)(base_pwm * 0.6f);
        else if (remaining >  5.0f) pwm = (int)(base_pwm * 0.4f);
        else                        pwm = (int)(base_pwm * 0.3f);

        if (pwm < pwmMin) pwm = pwmMin;
        if (pwm > pwmMax) pwm = pwmMax;

        // -----------------------------
        // 4. Gyro PID
        // -----------------------------
        heading += gz_dps * dt;   // integrate gyro

        float error = -heading;   // target heading = 0
        integral += error * dt;

        // Anti-windup
        if (integral > 20.0f) integral = 20.0f;
        if (integral < -20.0f) integral = -20.0f;

        float derivative = (error - last_error) / dt;
        last_error = error;

        int heading_correction =
            (int)(HEADING_KP * error +
                  HEADING_KI * integral +
                  HEADING_KD * derivative);

        if (heading_correction > 2000)  heading_correction = 2000;
        if (heading_correction < -2000) heading_correction = -2000;

        // -----------------------------
        // 5. Motor Mixing (Direction-Aware Trim)
        // -----------------------------
        int left_cmd  = dir * pwm + dir * MOTOR_TRIM - heading_correction;
        int right_cmd = dir * pwm - dir * MOTOR_TRIM + heading_correction;

        // Saturate PWM
        if (left_cmd  > pwmMax)  left_cmd  = pwmMax;
        if (left_cmd  < -pwmMax) left_cmd  = -pwmMax;
        if (right_cmd > pwmMax)  right_cmd = pwmMax;
        if (right_cmd < -pwmMax) right_cmd = -pwmMax;

        set_left_motor(left_cmd);
        set_right_motor(right_cmd);

        HAL_Delay(10);
    }

    // -----------------------------
    // 6. Stop Motors (always stop when exiting loop)
    // -----------------------------
    Motor_stop();
}

// Forward-only obstacle stop (HC-SR04 faces forward)





void set_left_motor(int pwm)
{
    if (pwm > 0)
    {
        __HAL_TIM_SetCompare(&htim4, TIM_CHANNEL_3, pwm);
        __HAL_TIM_SetCompare(&htim4, TIM_CHANNEL_4, 0);
    }
    else if (pwm < 0)
    {
        __HAL_TIM_SetCompare(&htim4, TIM_CHANNEL_3, 0);
        __HAL_TIM_SetCompare(&htim4, TIM_CHANNEL_4, -pwm);
    }
    else
    {
        __HAL_TIM_SetCompare(&htim4, TIM_CHANNEL_3, 0);
        __HAL_TIM_SetCompare(&htim4, TIM_CHANNEL_4, 0);
    }
}

void set_right_motor(int pwm)
{
    if (pwm > 0)
    {
        __HAL_TIM_SetCompare(&htim1, TIM_CHANNEL_4, pwm);
        __HAL_TIM_SetCompare(&htim1, TIM_CHANNEL_3, 0);
    }
    else if (pwm < 0)
    {
        __HAL_TIM_SetCompare(&htim1,
TIM_CHANNEL_4, 0);
        __HAL_TIM_SetCompare(&htim1, TIM_CHANNEL_3, -pwm);
    }
    else
    {
        __HAL_TIM_SetCompare(&htim1, TIM_CHANNEL_4, 0);
        __HAL_TIM_SetCompare(&htim1, TIM_CHANNEL_3, 0);
    }
}


static void Drive_Straight_ToCM_ForwardWorks(float target_cm, int base_pwm)// forwards only, alongside gyro
{
    Steering_ToUS(0);       // center steering
    reset_encoders();       // reset distance counters

    const int dir = (target_cm >= 0.0f) ? +1 : -1;
    const float STOP_TOL_CM = fmaxf(2.0f, fabsf(target_cm) * 0.06f);

    // Gyro-based PID gains for heading
    const float Kp_gyro = 40.0f;
    const float Ki_gyro = 4.0f;
    const float Kd_gyro = 3.5f;

    // Optional small encoder balance gain (to complement gyro)
    const float Kp_enc = 15.0f;

    float heading = 0.0f;
    float target_heading = 0.0f;
    float integral = 0.0f;
    float last_error = 0.0f;
    float gz_filtered = 0.0f;
    uint8_t initialized = 0;

    uint32_t last_time = HAL_GetTick();
    uint32_t lastPrint = 0;

    while (1)
    {
        // --- Distance tracking from encoders ---
        int32_t dL = left_ticks_signed();
        int32_t dR = right_ticks_signed();
        float normL = (float)dL / COUNTS_PER_CM_L;
        float normR = (float)dR / COUNTS_PER_CM_R;
        float done_cm = (normL + normR) * 0.5f * dir;  // average distance
        float left_cm = target_cm - done_cm;

        if (fabsf(left_cm) <= STOP_TOL_CM) break;

        // --- Smooth braking curve ---
        int pwm = base_pwm;
        if      (fabsf(left_cm) > 30.0f) pwm = base_pwm;
        else if (fabsf(left_cm) > 10.0f) pwm = (int)(base_pwm * 0.60f);
        else if (fabsf(left_cm) >  3.0f) pwm = (int)(base_pwm * 0.35f);
        else                             pwm = (int)(base_pwm * 0.25f);

        if (pwm < pwmMin) pwm = pwmMin;
        if (pwm > pwmMax) pwm = pwmMax;

        // --- Encoder-based lateral correction ---
        float e_enc = (normL - normR) * dir;  // positive if left wheel ahead
        int corr_enc = (int)(Kp_enc * e_enc);
        if (corr_enc > 1200) corr_enc = 1200;
        if (corr_enc < -1200) corr_enc = -1200;

        // --- Gyro-based heading PID ---
        uint32_t now = HAL_GetTick();
        float dt = (now - last_time) / 1000.0f;
        if (dt <= 0) dt = 0.001f;
        last_time = now;

        if (!initialized) { target_heading = heading; initialized = 1; }

        gz_filtered = 0.9f * gz_filtered + 0.1f * gz_dps;
        heading += gz_filtered * dt;

        float err = target_heading - heading;
        integral += err * dt;
        float deriv = (err - last_error) / dt;
        last_error = err;

        int corr_gyro = (int)(Kp_gyro * err + Ki_gyro * integral + Kd_gyro * deriv);
        if (corr_gyro > 2000) corr_gyro = 2000;
        if (corr_gyro < -2000) corr_gyro = -2000;

        // --- PWM commands ---
        int trim = (dir > 0) ? PWM_TRIM : -PWM_TRIM;
        int left_cmd  = pwm + trim - corr_enc + corr_gyro;
        int right_cmd = pwm - trim + corr_enc - corr_gyro;

        // Saturate PWM
        if (left_cmd  > pwmMax)  left_cmd  = pwmMax;
        if (left_cmd  < pwmMin)  left_cmd  = pwmMin;
        if (right_cmd > pwmMax)  right_cmd = pwmMax;
        if (right_cmd < pwmMin)  right_cmd = pwmMin;

        // --- Apply motor commands ---
        if (dir > 0) {
            __HAL_TIM_SetCompare(&htim4, TIM_CHANNEL_3, left_cmd);
            __HAL_TIM_SetCompare(&htim4, TIM_CHANNEL_4, 0);
            __HAL_TIM_SetCompare(&htim1, TIM_CHANNEL_3, 0);
            __HAL_TIM_SetCompare(&htim1, TIM_CHANNEL_4, right_cmd);
        } else {
            __HAL_TIM_SetCompare(&htim4, TIM_CHANNEL_3, 0);
            __HAL_TIM_SetCompare(&htim4, TIM_CHANNEL_4, left_cmd);
            __HAL_TIM_SetCompare(&htim1, TIM_CHANNEL_3, right_cmd);
            __HAL_TIM_SetCompare(&htim1, TIM_CHANNEL_4, 0);
        }

        if (now - lastPrint >= 100) {
            lastPrint = now;
            printf("STRAIGHT cm_done:%.1f cm_left:%.1f pwm:%d L:%d R:%d ENC:%d %d\r\n",
                   done_cm, left_cm, pwm, left_cmd, right_cmd, (int)dL, (int)dR);
        }

        HAL_Delay(10);
    }

    Motor_stop();
}

static void Drive_Straight_ToCM_old(float target_cm, int base_pwm)// not working backwards
{
    Steering_ToUS(0);       // center steering
    reset_encoders();       // reset distance counters

    const int dir = (target_cm >= 0.0f) ? +1 : -1;
    const float STOP_TOL_CM = fmaxf(2.0f, fabsf(target_cm) * 0.06f);

    // Gyro PID gains
    const float Kp_gyro = 40.0f;
    const float Ki_gyro = 4.0f;
    const float Kd_gyro = 3.5f;

    // Optional small encoder lateral gain
    const float Kp_enc = 15.0f;

    float heading = 0.0f;
    float target_heading = 0.0f;
    float integral = 0.0f;
    float last_error = 0.0f;
    float gz_filtered = 0.0f;
    uint8_t initialized = 0;

    uint32_t last_time = HAL_GetTick();
    uint32_t lastPrint = 0;

    while (1)
    {
        // --- Encoder distance ---
        int32_t dL = left_ticks_signed();
        int32_t dR = right_ticks_signed();
        float normL = (float)dL / COUNTS_PER_CM_L;
        float normR = (float)dR / COUNTS_PER_CM_R;

        // Average distance traveled (always positive magnitude)
        float done_cm = (normL + normR) * 0.5f;

        // Compute remaining distance depending on direction
        float left_cm = (dir > 0) ? (target_cm - done_cm) : (target_cm + done_cm);

        // Stop if within tolerance
        if (fabsf(left_cm) <= STOP_TOL_CM) break;

        // --- Smooth braking PWM ---
        int pwm = base_pwm;
        if      (fabsf(left_cm) > 30.0f) pwm = base_pwm;
        else if (fabsf(left_cm) > 10.0f) pwm = (int)(base_pwm * 0.60f);
        else if (fabsf(left_cm) >  3.0f) pwm = (int)(base_pwm * 0.35f);
        else                             pwm = (int)(base_pwm * 0.25f);

        if (pwm < pwmMin) pwm = pwmMin;
        if (pwm > pwmMax) pwm = pwmMax;

        // --- Encoder lateral correction ---
        float e_enc = (normL - normR);   // positive if left ahead
        int corr_enc = (int)(Kp_enc * e_enc);
        if (corr_enc > 1200) corr_enc = 1200;
        if (corr_enc < -1200) corr_enc = -1200;

        // --- Gyro heading PID ---
        uint32_t now = HAL_GetTick();
        float dt = (now - last_time) / 1000.0f;
        if (dt <= 0) dt = 0.001f;
        last_time = now;

        if (!initialized) { target_heading = heading; initialized = 1; }

        // Low-pass filter gyro rate
        gz_filtered = 0.9f * gz_filtered + 0.1f * gz_dps;
        heading += gz_filtered * dt;

        float err = target_heading - heading;
        integral += err * dt;
        float deriv = (err - last_error) / dt;
        last_error = err;

        int corr_gyro = (int)(Kp_gyro * err + Ki_gyro * integral + Kd_gyro * deriv);
        if (corr_gyro > 2000) corr_gyro = 2000;
        if (corr_gyro < -2000) corr_gyro = -2000;

        // --- PWM commands ---
        int trim = (dir > 0) ? PWM_TRIM : -PWM_TRIM;
        int left_cmd  = pwm + trim - corr_enc + corr_gyro;
        int right_cmd = pwm - trim + corr_enc - corr_gyro;

        // Saturate PWM
        if (left_cmd  > pwmMax)  left_cmd  = pwmMax;
        if (left_cmd  < pwmMin)  left_cmd  = pwmMin;
        if (right_cmd > pwmMax)  right_cmd = pwmMax;
        if (right_cmd < pwmMin)  right_cmd = pwmMin;

        // --- Apply motor PWM ---
        if (dir > 0) {
            __HAL_TIM_SetCompare(&htim4, TIM_CHANNEL_3, left_cmd);
            __HAL_TIM_SetCompare(&htim4, TIM_CHANNEL_4, 0);
            __HAL_TIM_SetCompare(&htim1, TIM_CHANNEL_3, 0);
            __HAL_TIM_SetCompare(&htim1, TIM_CHANNEL_4, right_cmd);
        } else {
            __HAL_TIM_SetCompare(&htim4, TIM_CHANNEL_3, 0);
            __HAL_TIM_SetCompare(&htim4, TIM_CHANNEL_4, left_cmd);
            __HAL_TIM_SetCompare(&htim1, TIM_CHANNEL_3, right_cmd);
            __HAL_TIM_SetCompare(&htim1, TIM_CHANNEL_4, 0);
        }

        // --- Debug print ---
        if (now - lastPrint >= 100) {
            lastPrint = now;
            printf("STRAIGHT cm_done:%.1f cm_left:%.1f pwm:%d L:%d R:%d ENC:%d %d\r\n",
                   (dir > 0 ? done_cm : -done_cm), left_cm, pwm, left_cmd, right_cmd, (int)dL, (int)dR);
        }

        HAL_Delay(10);
    }

    Motor_stop();
}


// HCSR04 (Ultrasonic Sensor) Reading Function
uint32_t HCSR04_Read(void)
{
    uint32_t start_tick, stop_tick, pulse_length;

    // --- 1. Send 10 µs pulse on TRIG ---
    HAL_GPIO_WritePin(GPIOB, GPIO_PIN_14, GPIO_PIN_SET);
    for (volatile int i = 0; i < 300; i++); // ~10 µs delay @168 MHz
    HAL_GPIO_WritePin(GPIOB, GPIO_PIN_14, GPIO_PIN_RESET);

    // --- 2. Wait for ECHO rising edge ---
    while (HAL_GPIO_ReadPin(GPIOC, GPIO_PIN_9) == GPIO_PIN_RESET);

    start_tick = DWT->CYCCNT;

    // --- 3. Wait for ECHO falling edge ---
    while (HAL_GPIO_ReadPin(GPIOC, GPIO_PIN_9) == GPIO_PIN_SET);

    stop_tick = DWT->CYCCNT;

    // --- 4. Compute pulse length ---
    pulse_length = stop_tick - start_tick;

    // Convert to µs (SystemCoreClock = 168 MHz → 1 tick = 1/168 MHz = 5.95 ns)
    uint32_t time_us = pulse_length / (SystemCoreClock / 1000000);

    // Distance (cm) = (time_us * 0.0343) / 2
    return (uint32_t)((time_us * 343) / 20000);  // optimized integer math
}


/* ----------------------- Gyro-based turning ------------------------ */
static float heading_deg = 0.0f;  // integrated yaw (deg)
static void Gyro_ResetHeading(void) { heading_deg = 0.0f; }

static void Gyro_UpdateFromIMU(float dt_s) {
  // Read IMU quickly and integrate gz (deg/s) to heading
  int16_t ax, ay, az, gx, gy, gz;
  ICM20948_ReadRaw(&ax, &ay, &az, &gx, &gy, &gz);
  gz_dps = (float)gz / 131.0f;   // ±250 dps scale
  heading_deg += gz_dps * dt_s;  // integrate
}

static float Gyro_GetHeadingDeg(void) { return heading_deg; }

/* Drive forward with fixed steering until we rotate target_deg */
static float Drive_Turn_Angle(float target_deg, int steer_deg, int base_pwm)
{
    if (target_deg < 0)
        target_deg = -target_deg;

    Gyro_ResetHeading();
    reset_encoders();

    if (steer_deg > 0) {
        Steering_ToUS_original(45);
    } else {
        Steering_ToUS_Fixed(steer_deg);
    }

    HAL_Delay(80);

    const float STOP_TOL_DEG = fmaxf(0.2f, target_deg * 0.01f);
    const int CRUISE_PWM = base_pwm;
    const int SLOW_PWM   = (int)(base_pwm * 0.55f);

    uint32_t last = HAL_GetTick();
    uint32_t lastPrint = 0;

    while (1)
    {
        uint32_t now = HAL_GetTick();
        float dt = (now - last) / 1000.0f;
        if (dt <= 0) dt = 0.001f;
        last = now;

        Gyro_UpdateFromIMU(dt);

        float heading = -Gyro_GetHeadingDeg();   // signed
        float turned = fabsf(heading);
        float remaining = target_deg - turned;
        /*
        uint32_t distance_cm = HCSR04_Read();
        if (distance_cm <= 20)
        {
            Motor_stop();
            HAL_GPIO_WritePin(GPIOA, GPIO_PIN_8, GPIO_PIN_SET);
            HAL_Delay(500);
            HAL_GPIO_WritePin(GPIOA, GPIO_PIN_8, GPIO_PIN_RESET);
            OLED_ShowString(0, 40, (uint8_t*)"Obstacle detected!");
            break;
        }
		*/
        if (remaining <= STOP_TOL_DEG)
            break;

        int pwm = (remaining > 25.0f) ? CRUISE_PWM : SLOW_PWM;
        if (pwm < 260) pwm = 260;

        Motor_drive_turn(pwm, steer_deg, 1);

        if (now - lastPrint >= 100)
        {
            lastPrint = now;

            OLED_Clear();

            char buf[20];

            snprintf(buf, sizeof(buf), "Target:%.1f", target_deg);
            OLED_ShowString(0, 0, (uint8_t*)buf);

            snprintf(buf, sizeof(buf), "Turned:%.1f", turned);
            OLED_ShowString(0, 16, (uint8_t*)buf);

            snprintf(buf, sizeof(buf), "Remain:%.1f", remaining);
            OLED_ShowString(0, 28, (uint8_t*)buf);

            snprintf(buf, sizeof(buf), "Steer:%d", steer_deg);
            OLED_ShowString(0, 40, (uint8_t*)buf);

            OLED_Refresh_Gram();

            printf("TURN hdg:%.1f rem:%.1f steer:%d pwm:%d | dL:%ld dR:%ld\r\n",
                   heading,
                   remaining,
                   steer_deg,
                   pwm,
                   (long)left_ticks(),
                   (long)right_ticks());
        }

        HAL_Delay(10);
    }

    Motor_stop();
    HAL_Delay(80);

    // signed final angle
    float final_angle = -Gyro_GetHeadingDeg();

    float arc_len = cm_travelled();

    float theta_rad = fabsf(final_angle) * (3.14159f / 180.0f);

    float turn_radius = 0.0f;

    if (theta_rad > 0.001f){
    	turn_radius = arc_len / theta_rad;
    }

    OLED_Clear();

    char buf[20];

    //OLED_ShowString(0, 0, (uint8_t*)"TURN DONE");

    snprintf(buf, sizeof(buf), "Target:%.1f", target_deg);
    OLED_ShowString(0, 0, (uint8_t*)buf);

    snprintf(buf, sizeof(buf), "Final:%.1f", final_angle);
    OLED_ShowString(0, 16, (uint8_t*)buf);

    snprintf(buf, sizeof(buf), "Error:%.1f", fabsf(final_angle) - target_deg);
    OLED_ShowString(0, 28, (uint8_t*)buf);

    snprintf(buf, sizeof(buf), "Rad:%.1fcm", turn_radius);
    OLED_ShowString(0, 40, (uint8_t*)buf);

    OLED_Refresh_Gram();

    Steering_ToUS(0);

    printf("[STOP TURN] hdg:%.1f target:%.1f steer:%d | dL:%ld dR:%ld\r\n",
           final_angle,
           target_deg,
           steer_deg,
           (long)left_ticks(),
           (long)right_ticks());

    Gyro_ResetHeading();
    return final_angle;
}

static float Drive_Turn_AngleBW(float target_deg, int steer_deg, int base_pwm)
{
    if (target_deg < 0)
        target_deg = -target_deg;

    Gyro_ResetHeading();
    reset_encoders();

    if (steer_deg > 0) {
           Steering_ToUS_original(45);
        } else {
           Steering_ToUS_Fixed(steer_deg);
        }

    HAL_Delay(80);

    const float STOP_TOL_DEG = fmaxf(0.2f, target_deg * 0.01f);
    const int CRUISE_PWM = base_pwm;
    const int SLOW_PWM   = (int)(base_pwm * 0.55f);

    uint32_t last = HAL_GetTick();
    uint32_t lastPrint = 0;
    uint32_t start_time = HAL_GetTick();

    while (1)
    {
        uint32_t now = HAL_GetTick();
        float dt = (now - last) / 1000.0f;
        if (dt <= 0) dt = 0.001f;
        last = now;

        Gyro_UpdateFromIMU(dt);

        float heading = -Gyro_GetHeadingDeg();   // signed
        float turned = fabsf(heading);
        float remaining = target_deg - turned;

        if (remaining <= STOP_TOL_DEG)
            break;

        if (now - start_time > 7000)
        {
            OLED_Clear();
            OLED_ShowString(0, 0, (uint8_t*)"TURN TIMEOUT!");
            OLED_Refresh_Gram();
            break;
        }

        int pwm = (remaining > 25.0f) ? CRUISE_PWM : SLOW_PWM;
        if (pwm < 260) pwm = 260;

        Motor_drive_turn(pwm, steer_deg, -1);

        if (now - lastPrint >= 100)
        {
            lastPrint = now;

            OLED_Clear();

            char buf[20];

            snprintf(buf, sizeof(buf), "Target:%.1f", target_deg);
            OLED_ShowString(0, 0, (uint8_t*)buf);

            snprintf(buf, sizeof(buf), "Turned:%.1f", turned);
            OLED_ShowString(0, 16, (uint8_t*)buf);

            snprintf(buf, sizeof(buf), "Remain:%.1f", remaining);
            OLED_ShowString(0, 28, (uint8_t*)buf);

            snprintf(buf, sizeof(buf), "Steer:%d", steer_deg);
            OLED_ShowString(0, 40, (uint8_t*)buf);

            OLED_Refresh_Gram();

            printf("BW TURN hdg:%.1f rem:%.1f steer:%d pwm:%d | dL:%ld dR:%ld\r\n",
                   heading,
                   remaining,
                   steer_deg,
                   pwm,
                   (long)left_ticks(),
                   (long)right_ticks());
        }

        HAL_Delay(10);
    }

    Motor_stop();
    HAL_Delay(80);

    float final_angle = -Gyro_GetHeadingDeg();

    float arc_len = cm_travelled();

	float theta_rad = fabsf(final_angle) * (3.14159f / 180.0f);

	float turn_radius = 0.0f;

	if (theta_rad > 0.001f){
		turn_radius = arc_len / theta_rad;
	}

    OLED_Clear();

    char buf[20];

    //OLED_ShowString(0, 0, (uint8_t*)"BW TURN DONE");

    snprintf(buf, sizeof(buf), "Target:%.1f", target_deg);
    OLED_ShowString(0, 0, (uint8_t*)buf);

    snprintf(buf, sizeof(buf), "Final:%.1f", final_angle);
    OLED_ShowString(0, 16, (uint8_t*)buf);

    snprintf(buf, sizeof(buf), "Error:%.1f", fabsf(final_angle) - target_deg);
    OLED_ShowString(0, 28, (uint8_t*)buf);

    snprintf(buf, sizeof(buf), "Rad:%.1fcm", turn_radius);
    OLED_ShowString(0, 40, (uint8_t*)buf);

    OLED_Refresh_Gram();

    Steering_ToUS(0);

    printf("[STOP BW TURN] hdg:%.1f target:%.1f steer:%d | dL:%ld dR:%ld\r\n",
           final_angle,
           target_deg,
           steer_deg,
           (long)left_ticks(),
           (long)right_ticks());

    Gyro_ResetHeading();
    return final_angle;
}

// Left and right wheel tick checking
static void Debug_Show_WheelTicks(void)
{
    static int32_t lastL = 0;
    static int32_t lastR = 0;

    int32_t L = left_ticks_signed();
    int32_t R = right_ticks_signed();

    int32_t dL = L - lastL;
    int32_t dR = R - lastR;

    lastL = L;
    lastR = R;

    OLED_Clear();

    char buf[20];

    snprintf(buf, sizeof(buf), "L:%ld", (long)L);
    OLED_ShowString(0, 0, (uint8_t*)buf);

    snprintf(buf, sizeof(buf), "R:%ld", (long)R);
    OLED_ShowString(0, 12, (uint8_t*)buf);

    snprintf(buf, sizeof(buf), "dL:%ld", (long)dL);
    OLED_ShowString(0, 24, (uint8_t*)buf);

    snprintf(buf, sizeof(buf), "dR:%ld", (long)dR);
    OLED_ShowString(0, 36, (uint8_t*)buf);

    OLED_Refresh_Gram();
}
static float Apply_Straight_Deviation_To_Turn(float requested_signed_deg, int backward)
{
    float corrected;

    if (!backward)
    {
        // forward turns
        corrected = requested_signed_deg + g_straight_dev_deg;
    }
    else
    {
        // backward turns (sign flips)
        corrected = requested_signed_deg - g_straight_dev_deg;
    }

    //if (corrected > 360.0f) corrected = 360.0f;
    //if (corrected < -360.0f) corrected = -360.0f;

    if (requested_signed_deg > 0.0f && corrected < 1.0f)
        corrected = 1.0f;

    if (requested_signed_deg < 0.0f && corrected > -1.0f)
        corrected = -1.0f;

    return corrected;
}

static float Apply_Turn_Compensation(float requested_signed_deg, int backward)
{
    float corrected = Apply_Straight_Deviation_To_Turn(requested_signed_deg, backward);
    corrected += g_turn_residual_deg;

    // Keep command direction aligned with the requested turn.
    if (requested_signed_deg > 0.0f && corrected < 1.0f)
        corrected = 1.0f;

    if (requested_signed_deg < 0.0f && corrected > -1.0f)
        corrected = -1.0f;

    return corrected;
}

static void Update_Turn_Residual(float requested_signed_deg, float actual_signed_deg)
{
    float prev = g_turn_residual_deg;
    float turn_error = requested_signed_deg - actual_signed_deg;

    g_turn_residual_deg = clampf(
        g_turn_residual_deg + turn_error,
        TURN_RESIDUAL_MIN_DEG,
        TURN_RESIDUAL_MAX_DEG
    );

    printf("TURN_RES update prev=%.2f err=%.2f new=%.2f req=%.2f act=%.2f\r\n",
           prev, turn_error, g_turn_residual_deg, requested_signed_deg, actual_signed_deg);
}




static float Execute_Signed_Turn(float cmd_signed_deg, int base_pwm, int backward)
{
    if (cmd_signed_deg >= 0.0f)
    {
        return backward
            ? Drive_Turn_AngleBW(fabsf(cmd_signed_deg), +45, base_pwm)
            : Drive_Turn_Angle(fabsf(cmd_signed_deg), +45, base_pwm);
    }
    else
    {
        return backward
            ? Drive_Turn_AngleBW(fabsf(cmd_signed_deg), -30, base_pwm)
            : Drive_Turn_Angle(fabsf(cmd_signed_deg), -30, base_pwm);
    }
}
void process_command(char *cmd)
{
    char *t = cmd;
    while (*t == '\r' || *t == '\n') t++;
    Display_UART_Command(t);

    // 1) Straight motion
    if (strncmp(t, "SF", 2) == 0) {
        int cm = atoi(t + 2);
        ACK("SF%03d", cm);
        Steering_ToUS(0);
        Drive_Straight_ToCM((float)cm, 2200);   // blocks until done
        DONE("SF done,cm=%.1f", cm_travelled_signed());
        return;
    }
    if (strncmp(t, "SB", 2) == 0) {
        int cm = atoi(t + 2);
        ACK("SB%03d", cm);
        Steering_ToUS(0);
        Drive_Straight_ToCM(-(float)cm, 2200);      // blocks until done
        DONE("SB done,cm=%.1f", cm_travelled_signed());
        return;
    }

    if (strncmp(t, "RS", 2) == 0) {
        g_straight_dev_deg = 0.0f;
        g_turn_residual_deg = 0.0f;

        ACK("RS");

        printf("STRAIGHT_DEV reset to 0\r\n");
        printf("TURN_RES reset to 0\r\n");

        DONE("RS done,dev=%.2f", g_straight_dev_deg);
        return;
    }

    // 2) Turns with global signed residual compensation
    if (strncmp(t, "TL", 2) == 0) {
    	float req_signed = -(float)atoi(t + 2);
    	float cmd_signed = Apply_Turn_Compensation(req_signed, 0);
        ACK("TL%03d", (int)fabsf(req_signed));

        float actual_signed = Execute_Signed_Turn(cmd_signed, 2400, 0);
        Update_Turn_Residual(req_signed, actual_signed);
        DONE("TR done,req=%.1f cmd=%.1f act=%.1f sdev=%.2f",
             req_signed, cmd_signed, actual_signed, g_straight_dev_deg);
        return;
    }

    if (strncmp(t, "TR", 2) == 0) {
        float req_signed = +(float)atoi(t + 2);
        float cmd_signed = Apply_Turn_Compensation(req_signed, 0);

        ACK("TR%03d", (int)fabsf(req_signed));

        float actual_signed = Execute_Signed_Turn(cmd_signed, 2100, 0);
        Update_Turn_Residual(req_signed, actual_signed);

        DONE("TR done,req=%.1f cmd=%.1f act=%.1f sdev=%.2f",
             req_signed, cmd_signed, actual_signed, g_straight_dev_deg);
        return;
    }

    if (strncmp(t, "BTL", 3) == 0) {
    	float req_signed = -(float)atoi(t + 3);
    	float cmd_signed = Apply_Turn_Compensation(req_signed, 1);
        ACK("BTL%03d", (int)fabsf(req_signed));

        float actual_signed = Execute_Signed_Turn(cmd_signed, 2200, 1);
        Update_Turn_Residual(req_signed, actual_signed);
        DONE("TR done,req=%.1f cmd=%.1f act=%.1f sdev=%.2f",
             req_signed, cmd_signed, actual_signed, g_straight_dev_deg);
    }

    if (strncmp(t, "BTR", 3) == 0) {
    	float req_signed = +(float)atoi(t + 3);
    	float cmd_signed = Apply_Turn_Compensation(req_signed, 1);

        ACK("BTR%03d", (int)fabsf(req_signed));

        float actual_signed = Execute_Signed_Turn(cmd_signed, 2200, 1);
        Update_Turn_Residual(req_signed, actual_signed);
        DONE("TR done,req=%.1f cmd=%.1f act=%.1f sdev=%.2f",
             req_signed, cmd_signed, actual_signed, g_straight_dev_deg);
    }
    if (strncmp(t, "PING20", 6)== 0){
    	ACK("PING20");
    	Steering_ToUS(0);
    	Drive_Straight_ToCM((float)200, 2200);      // blocks until done
    	DONE("PING20 done, cm=%.1f", cm_travelled_signed());
    	return;
    }

    // 3) Snapshot markers
    if (strncmp(t, "SNAP", 4) == 0 && t[4] != '\0') {
            int snap_num = atoi(t + 4);
            Motor_stop();
            ACK("SNAP%d", snap_num);
            DONE("SNAP%d done", snap_num);
            return;
      }
    /*if (strncmp(t, "SNAP2", 5) == 0) {
        Motor_stop();
        ACK("SNAP2");
        DONE("SNAP2 done");
        return;
    }*/

    // 4) Stop / End
    if (strncmp(t, "STOP", 4) == 0) {
        Motor_stop();
        ACK("STOP");
        DONE("stop");
        return;
    }
    if (strncmp(t, "FIN", 3) == 0) {
        Motor_stop();
        g_straight_dev_deg = 0.0f;
        g_turn_residual_deg = 0.0f;
        printf("TURN_RES reset to 0\r\n");
        ACK("FIN");
        DONE("end sequence");
        return;
    }

    // Unknown command
    ERR("unknown,%s", t);
}


/* ---------------------- Minimal UART printf ----------------------- */
int _write(int file, char *ptr, int len) {
  HAL_UART_Transmit(&huart3, (uint8_t *)ptr, len, HAL_MAX_DELAY);
  return len;
}
static uint8_t EmergencyStop_Usonic(void)
{
    static uint32_t last_check = 0;
    uint32_t now = HAL_GetTick();

    // check at 10 Hz (every 100 ms) so you don’t spam the sensor
    if (now - last_check < 100) return 0;
    last_check = now;

    uint32_t distance_cm = HCSR04_Read();

    if (distance_cm > 0 && distance_cm <= 20)
    {
        Motor_stop();
        Steering_ToUS(0);

        OLED_Clear();
        OLED_ShowString(0, 0, (uint8_t*)"EMERGENCY STOP");
        char buf[20];
        snprintf(buf, sizeof(buf), "Dist:%lu cm", (unsigned long)distance_cm);
        OLED_ShowString(0, 16, (uint8_t*)buf);
        OLED_ShowString(0, 32, (uint8_t*)"Obstacle!");
        OLED_Refresh_Gram();

        return 1; // stopped
    }
    return 0;
}

static void Test_Turn_Residual(void)
{
	process_command("SF100");
	HAL_Delay(1000);



}
/* =============================  main_  ============================= */
int main(void)
{
  HAL_Init();
  SystemClock_Config();

  MX_GPIO_Init();
  MX_TIM8_Init();
  MX_TIM2_Init();
  MX_USART2_UART_Init();
  MX_TIM1_Init();
  MX_USART3_UART_Init();
  MX_I2C2_Init();
  MX_TIM5_Init();
  MX_TIM4_Init();
  MX_TIM3_Init();
  MX_TIM11_Init();
  MX_TIM12_Init();
  MX_ADC1_Init();

  /* Peripherals start */
  HAL_TIM_Encoder_Start(&htim2, TIM_CHANNEL_ALL);  // Left encoder (TIM2)
  HAL_TIM_Encoder_Start(&htim5, TIM_CHANNEL_ALL);  // Right encoder (TIM5)
  MotorDrive_enable();                             // PWM for motors
  HAL_TIM_PWM_Start(&htim12, TIM_CHANNEL_2);       // Servo PWM

  OLED_Init();
  OLED_Clear();
  OLED_ShowString(0,0,(uint8_t*)"STM Car Ready");

  HAL_Delay(100);
  if (ICM20948_Detect() == 0 && ICM20948_Init() == 0) {
    OLED_ShowString(0, 12, (uint8_t*)"IMU OK");
  } else {
    OLED_ShowString(0, 12, (uint8_t*)"IMU FAIL");
  }
  OLED_Refresh_Gram();

  CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
  DWT->CYCCNT = 0;
  DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
  Steering_ToUS(30);
  Steering_ToUS(0);
  reset_encoders();

 //Drive_Straight_ToCM((float)200, 2200);
 //Encoder_Calibration_Test();
Test_Turn_Residual();
// This while loop is for testing ticks of wheels only
/*  while (1)
  {
      static int32_t lastL = 0;
      static int32_t lastR = 0;
      static uint32_t lastTime = 0;
      static uint32_t startTime = 0;

      uint32_t now = HAL_GetTick();

      // capture start time once
      if (startTime == 0)
          startTime = now;

      // stop after 3 seconds (safe test window)
      if (now - startTime > 3000)
      {
          Motor_stop();
      }

      // update OLED every 100 ms
      if (now - lastTime >= 100)
      {
          lastTime = now;

          int32_t L = left_ticks_signed();
          int32_t R = right_ticks_signed();

          int32_t dL = L - lastL;
          int32_t dR = R - lastR;

          lastL = L;
          lastR = R;

          float ratio = 0.0f;
          if (dR != 0)
              ratio = (float)dL / (float)dR;

          OLED_Clear();

          char buf[20];

          snprintf(buf, sizeof(buf), "L:%ld", (long)L);
          OLED_ShowString(0, 0, (uint8_t*)buf);

          snprintf(buf, sizeof(buf), "R:%ld", (long)R);
          OLED_ShowString(0, 12, (uint8_t*)buf);

          snprintf(buf, sizeof(buf), "dL:%ld", (long)dL);
          OLED_ShowString(0, 24, (uint8_t*)buf);

          snprintf(buf, sizeof(buf), "dR:%ld", (long)dR);
          OLED_ShowString(0, 36, (uint8_t*)buf);

          snprintf(buf, sizeof(buf), "ratio:%.2f", ratio);
          OLED_ShowString(0, 48, (uint8_t*)buf);

          OLED_Refresh_Gram();
      }
  }*/

  // this is main while loop
  while (1)
    {
	   // Drive_Straight_ToCM_USONIC((float)100, 2200);
     //Drive_Straight_ToCM(-(float)100, 2200);
 	  //Drive_Turn_Angle(90.0f, -25, 2200);//
 	  //HAL_Delay(500);
      //Drive_Turn_AngleBW(90.0f, -25, 2200);
      //HAL_Delay(500);
      //Drive_Turn_Angle(90.0f, +45, 2200);
 	  //HAL_Delay(500);
      //Drive_Turn_AngleBW(90.0f, +45, 2200);
      //HAL_Delay(500);
//	    Drive_Turn_Angle(30.0f, -30çç, 2200);

//	    Drive_Turn_AngleBW(30.0f, +55, 2200);

	  //iteration 1
	/*   Drive_Turn_Angle(30.0f, +30, 2200);
	    HAL_Delay(50);
	    Drive_Turn_Angle(60.0f, +30, 2200);
	    HAL_Delay(50);
	    //iteration 2
	    Drive_Turn_Angle(90.0f, +30, 2200);
		HAL_Delay(50);
		Drive_Turn_Angle(180.0f, +30, 2200);
		HAL_Delay(50);*/
		/*//iteration 3
		Drive_Turn_Angle(90.0f, +30, 2200);
		HAL_Delay(50);
		Drive_Turn_Angle(180.0f, -30, 2200);
		HAL_Delay(50);
		//iteration 4
		Drive_Turn_Angle(90.0f, +30, 2200);
		HAL_Delay(50);
		Drive_Turn_Angle(180.0f, -30, 2200);
		HAL_Delay(50);*/
	  // ✅ Emergency stop check always runs


	      uint8_t ch;
	      if (HAL_UART_Receive(&huart3, &ch, 1, 10) == HAL_OK)
	      {
	          if (ch == '\n' || ch == '\r')
	          {
	              if (cmd_index > 0)
	              {
	                  cmd_buf[cmd_index] = '\0';
	                  process_command(cmd_buf);
	                  cmd_index = 0;
	              }
	          }
	          else if (cmd_index < CMD_BUF_LEN - 1)
	          {
	              cmd_buf[cmd_index++] = ch;
	          }
	          else
	          {
	              cmd_index = 0;
	          }
	      }
    }

	      // (optional) small delay so CPU isn't 100%
	      HAL_Delay(1);
}


void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Configure the main internal regulator output voltage
  */
  __HAL_RCC_PWR_CLK_ENABLE();
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSE;
  RCC_OscInitStruct.HSEState = RCC_HSE_ON;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSE;
  RCC_OscInitStruct.PLL.PLLM = 8;
  RCC_OscInitStruct.PLL.PLLN = 168;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV2;
  RCC_OscInitStruct.PLL.PLLQ = 4;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV2;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_2) != HAL_OK)
  {
    Error_Handler();
  }
}

/**
  * @brief ADC1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_ADC1_Init(void)
{

  /* USER CODE BEGIN ADC1_Init 0 */

  /* USER CODE END ADC1_Init 0 */

  ADC_ChannelConfTypeDef sConfig = {0};

  /* USER CODE BEGIN ADC1_Init 1 */

  /* USER CODE END ADC1_Init 1 */

  /** Configure the global features of the ADC (Clock, Resolution, Data Alignment and number of conversion)
  */
  hadc1.Instance = ADC1;
  hadc1.Init.ClockPrescaler = ADC_CLOCK_SYNC_PCLK_DIV2;
  hadc1.Init.Resolution = ADC_RESOLUTION_12B;
  hadc1.Init.ScanConvMode = DISABLE;
  hadc1.Init.ContinuousConvMode = DISABLE;
  hadc1.Init.DiscontinuousConvMode = DISABLE;
  hadc1.Init.ExternalTrigConvEdge = ADC_EXTERNALTRIGCONVEDGE_NONE;
  hadc1.Init.ExternalTrigConv = ADC_SOFTWARE_START;
  hadc1.Init.DataAlign = ADC_DATAALIGN_RIGHT;
  hadc1.Init.NbrOfConversion = 1;
  hadc1.Init.DMAContinuousRequests = DISABLE;
  hadc1.Init.EOCSelection = ADC_EOC_SINGLE_CONV;
  if (HAL_ADC_Init(&hadc1) != HAL_OK)
  {
    Error_Handler();
  }

  /** Configure for the selected ADC regular channel its corresponding rank in the sequencer and its sample time.
  */
  sConfig.Channel = ADC_CHANNEL_4;
  sConfig.Rank = 1;
  sConfig.SamplingTime = ADC_SAMPLETIME_3CYCLES;
  if (HAL_ADC_ConfigChannel(&hadc1, &sConfig) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN ADC1_Init 2 */

  /* USER CODE END ADC1_Init 2 */

}

/**
  * @brief I2C2 Initialization Function
  * @param None
  * @retval None
  */
static void MX_I2C2_Init(void)
{

  /* USER CODE BEGIN I2C2_Init 0 */

  /* USER CODE END I2C2_Init 0 */

  /* USER CODE BEGIN I2C2_Init 1 */

  /* USER CODE END I2C2_Init 1 */
  hi2c2.Instance = I2C2;
  hi2c2.Init.ClockSpeed = 100000;
  hi2c2.Init.DutyCycle = I2C_DUTYCYCLE_2;
  hi2c2.Init.OwnAddress1 = 0;
  hi2c2.Init.AddressingMode = I2C_ADDRESSINGMODE_7BIT;
  hi2c2.Init.DualAddressMode = I2C_DUALADDRESS_DISABLE;
  hi2c2.Init.OwnAddress2 = 0;
  hi2c2.Init.GeneralCallMode = I2C_GENERALCALL_DISABLE;
  hi2c2.Init.NoStretchMode = I2C_NOSTRETCH_DISABLE;
  if (HAL_I2C_Init(&hi2c2) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN I2C2_Init 2 */

  /* USER CODE END I2C2_Init 2 */

}

/**
  * @brief TIM1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_TIM1_Init(void)
{

  /* USER CODE BEGIN TIM1_Init 0 */

  /* USER CODE END TIM1_Init 0 */

  TIM_ClockConfigTypeDef sClockSourceConfig = {0};
  TIM_MasterConfigTypeDef sMasterConfig = {0};
  TIM_OC_InitTypeDef sConfigOC = {0};
  TIM_BreakDeadTimeConfigTypeDef sBreakDeadTimeConfig = {0};

  /* USER CODE BEGIN TIM1_Init 1 */

  /* USER CODE END TIM1_Init 1 */
  htim1.Instance = TIM1;
  htim1.Init.Prescaler = 0;
  htim1.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim1.Init.Period = 7199;
  htim1.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim1.Init.RepetitionCounter = 0;
  htim1.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  if (HAL_TIM_Base_Init(&htim1) != HAL_OK)
  {
    Error_Handler();
  }
  sClockSourceConfig.ClockSource = TIM_CLOCKSOURCE_INTERNAL;
  if (HAL_TIM_ConfigClockSource(&htim1, &sClockSourceConfig) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_TIM_PWM_Init(&htim1) != HAL_OK)
  {
    Error_Handler();
  }
  sMasterConfig.MasterOutputTrigger = TIM_TRGO_RESET;
  sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
  if (HAL_TIMEx_MasterConfigSynchronization(&htim1, &sMasterConfig) != HAL_OK)
  {
    Error_Handler();
  }
  sConfigOC.OCMode = TIM_OCMODE_PWM1;
  sConfigOC.Pulse = 0;
  sConfigOC.OCPolarity = TIM_OCPOLARITY_HIGH;
  sConfigOC.OCNPolarity = TIM_OCNPOLARITY_HIGH;
  sConfigOC.OCFastMode = TIM_OCFAST_DISABLE;
  sConfigOC.OCIdleState = TIM_OCIDLESTATE_RESET;
  sConfigOC.OCNIdleState = TIM_OCNIDLESTATE_RESET;
  if (HAL_TIM_PWM_ConfigChannel(&htim1, &sConfigOC, TIM_CHANNEL_1) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_TIM_PWM_ConfigChannel(&htim1, &sConfigOC, TIM_CHANNEL_2) != HAL_OK)
  {
    Error_Handler();
  }
  sConfigOC.OCPolarity = TIM_OCPOLARITY_LOW;
  if (HAL_TIM_PWM_ConfigChannel(&htim1, &sConfigOC, TIM_CHANNEL_3) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_TIM_PWM_ConfigChannel(&htim1, &sConfigOC, TIM_CHANNEL_4) != HAL_OK)
  {
    Error_Handler();
  }
  sBreakDeadTimeConfig.OffStateRunMode = TIM_OSSR_DISABLE;
  sBreakDeadTimeConfig.OffStateIDLEMode = TIM_OSSI_DISABLE;
  sBreakDeadTimeConfig.LockLevel = TIM_LOCKLEVEL_OFF;
  sBreakDeadTimeConfig.DeadTime = 0;
  sBreakDeadTimeConfig.BreakState = TIM_BREAK_DISABLE;
  sBreakDeadTimeConfig.BreakPolarity = TIM_BREAKPOLARITY_HIGH;
  sBreakDeadTimeConfig.AutomaticOutput = TIM_AUTOMATICOUTPUT_DISABLE;
  if (HAL_TIMEx_ConfigBreakDeadTime(&htim1, &sBreakDeadTimeConfig) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN TIM1_Init 2 */

  /* USER CODE END TIM1_Init 2 */
  HAL_TIM_MspPostInit(&htim1);

}

/**
  * @brief TIM2 Initialization Function
  * @param None
  * @retval None
  */
static void MX_TIM2_Init(void) {
  TIM_Encoder_InitTypeDef sConfig = {0};
  TIM_MasterConfigTypeDef sMasterConfig = {0};
  htim2.Instance = TIM2;
  htim2.Init.Prescaler = 0;
  htim2.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim2.Init.Period = 65535;
  htim2.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim2.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  /* === IMPORTANT: true quadrature === */
  sConfig.EncoderMode = TIM_ENCODERMODE_TI12;
  sConfig.IC1Polarity = TIM_ICPOLARITY_RISING;
  sConfig.IC1Selection = TIM_ICSELECTION_DIRECTTI;
  sConfig.IC1Prescaler = TIM_ICPSC_DIV1;
  sConfig.IC1Filter = 0;
  sConfig.IC2Polarity = TIM_ICPOLARITY_RISING;
  sConfig.IC2Selection = TIM_ICSELECTION_DIRECTTI;
  sConfig.IC2Prescaler = TIM_ICPSC_DIV1;
  sConfig.IC2Filter = 0;
  HAL_TIM_Encoder_Init(&htim2, &sConfig);
  sMasterConfig.MasterOutputTrigger = TIM_TRGO_RESET;
  sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
  HAL_TIMEx_MasterConfigSynchronization(&htim2, &sMasterConfig);
}

static void MX_TIM5_Init(void) {
  TIM_Encoder_InitTypeDef sConfig = {0};
  TIM_MasterConfigTypeDef sMasterConfig = {0};
  htim5.Instance = TIM5;
  htim5.Init.Prescaler = 0;
  htim5.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim5.Init.Period = 65535;
  htim5.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim5.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  /* === IMPORTANT: true quadrature === */
  sConfig.EncoderMode = TIM_ENCODERMODE_TI12;
  sConfig.IC1Polarity = TIM_ICPOLARITY_RISING;
  sConfig.IC1Selection = TIM_ICSELECTION_DIRECTTI;
  sConfig.IC1Prescaler = TIM_ICPSC_DIV1;
  sConfig.IC1Filter = 0;
  sConfig.IC2Polarity = TIM_ICPOLARITY_RISING;
  sConfig.IC2Selection = TIM_ICSELECTION_DIRECTTI;
  sConfig.IC2Prescaler = TIM_ICPSC_DIV1;
  sConfig.IC2Filter = 0;
  HAL_TIM_Encoder_Init(&htim5, &sConfig);
  sMasterConfig.MasterOutputTrigger = TIM_TRGO_RESET;
  sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
  HAL_TIMEx_MasterConfigSynchronization(&htim5, &sMasterConfig);
}

/**
  * @brief TIM3 Initialization Function
  * @param None
  * @retval None
  */
static void MX_TIM3_Init(void)
{

  /* USER CODE BEGIN TIM3_Init 0 */

  /* USER CODE END TIM3_Init 0 */

  TIM_ClockConfigTypeDef sClockSourceConfig = {0};
  TIM_MasterConfigTypeDef sMasterConfig = {0};
  TIM_OC_InitTypeDef sConfigOC = {0};

  /* USER CODE BEGIN TIM3_Init 1 */

  /* USER CODE END TIM3_Init 1 */
  htim3.Instance = TIM3;
  htim3.Init.Prescaler = 720;
  htim3.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim3.Init.Period = 2000;
  htim3.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim3.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  if (HAL_TIM_Base_Init(&htim3) != HAL_OK)
  {
    Error_Handler();
  }
  sClockSourceConfig.ClockSource = TIM_CLOCKSOURCE_INTERNAL;
  if (HAL_TIM_ConfigClockSource(&htim3, &sClockSourceConfig) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_TIM_PWM_Init(&htim3) != HAL_OK)
  {
    Error_Handler();
  }
  sMasterConfig.MasterOutputTrigger = TIM_TRGO_RESET;
  sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
  if (HAL_TIMEx_MasterConfigSynchronization(&htim3, &sMasterConfig) != HAL_OK)
  {
    Error_Handler();
  }
  sConfigOC.OCMode = TIM_OCMODE_PWM1;
  sConfigOC.Pulse = 0;
  sConfigOC.OCPolarity = TIM_OCPOLARITY_HIGH;
  sConfigOC.OCFastMode = TIM_OCFAST_DISABLE;
  if (HAL_TIM_PWM_ConfigChannel(&htim3, &sConfigOC, TIM_CHANNEL_4) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN TIM3_Init 2 */

  /* USER CODE END TIM3_Init 2 */
  HAL_TIM_MspPostInit(&htim3);

}

/**
  * @brief TIM4 Initialization Function
  * @param None
  * @retval None
  */
static void MX_TIM4_Init(void)
{

  /* USER CODE BEGIN TIM4_Init 0 */

  /* USER CODE END TIM4_Init 0 */

  TIM_ClockConfigTypeDef sClockSourceConfig = {0};
  TIM_MasterConfigTypeDef sMasterConfig = {0};
  TIM_OC_InitTypeDef sConfigOC = {0};

  /* USER CODE BEGIN TIM4_Init 1 */

  /* USER CODE END TIM4_Init 1 */
  htim4.Instance = TIM4;
  htim4.Init.Prescaler = 0;
  htim4.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim4.Init.Period = 7199;
  htim4.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim4.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  if (HAL_TIM_Base_Init(&htim4) != HAL_OK)
  {
    Error_Handler();
  }
  sClockSourceConfig.ClockSource = TIM_CLOCKSOURCE_INTERNAL;
  if (HAL_TIM_ConfigClockSource(&htim4, &sClockSourceConfig) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_TIM_PWM_Init(&htim4) != HAL_OK)
  {
    Error_Handler();
  }
  sMasterConfig.MasterOutputTrigger = TIM_TRGO_RESET;
  sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
  if (HAL_TIMEx_MasterConfigSynchronization(&htim4, &sMasterConfig) != HAL_OK)
  {
    Error_Handler();
  }
  sConfigOC.OCMode = TIM_OCMODE_PWM1;
  sConfigOC.Pulse = 0;
  sConfigOC.OCPolarity = TIM_OCPOLARITY_LOW;
  sConfigOC.OCFastMode = TIM_OCFAST_DISABLE;
  if (HAL_TIM_PWM_ConfigChannel(&htim4, &sConfigOC, TIM_CHANNEL_3) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_TIM_PWM_ConfigChannel(&htim4, &sConfigOC, TIM_CHANNEL_4) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN TIM4_Init 2 */

  /* USER CODE END TIM4_Init 2 */
  HAL_TIM_MspPostInit(&htim4);

}

/**
  * @brief TIM5 Initialization Function
  * @param None
  * @retval None
  */

/**
  * @brief TIM8 Initialization Function
  * @param None
  * @retval None
  */
static void MX_TIM8_Init(void)
{

  /* USER CODE BEGIN TIM8_Init 0 */

  /* USER CODE END TIM8_Init 0 */

  TIM_ClockConfigTypeDef sClockSourceConfig = {0};
  TIM_MasterConfigTypeDef sMasterConfig = {0};

  /* USER CODE BEGIN TIM8_Init 1 */

  /* USER CODE END TIM8_Init 1 */
  htim8.Instance = TIM8;
  htim8.Init.Prescaler = 0;
  htim8.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim8.Init.Period = 7199;
  htim8.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim8.Init.RepetitionCounter = 0;
  htim8.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  if (HAL_TIM_Base_Init(&htim8) != HAL_OK)
  {
    Error_Handler();
  }
  sClockSourceConfig.ClockSource = TIM_CLOCKSOURCE_INTERNAL;
  if (HAL_TIM_ConfigClockSource(&htim8, &sClockSourceConfig) != HAL_OK)
  {
    Error_Handler();
  }
  sMasterConfig.MasterOutputTrigger = TIM_TRGO_RESET;
  sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
  if (HAL_TIMEx_MasterConfigSynchronization(&htim8, &sMasterConfig) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN TIM8_Init 2 */

  /* USER CODE END TIM8_Init 2 */

}

/**
  * @brief TIM11 Initialization Function
  * @param None
  * @retval None
  */
static void MX_TIM11_Init(void)
{

  /* USER CODE BEGIN TIM11_Init 0 */

  /* USER CODE END TIM11_Init 0 */

  TIM_OC_InitTypeDef sConfigOC = {0};

  /* USER CODE BEGIN TIM11_Init 1 */

  /* USER CODE END TIM11_Init 1 */
  htim11.Instance = TIM11;
  htim11.Init.Prescaler = 0;
  htim11.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim11.Init.Period = 7199;
  htim11.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim11.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  if (HAL_TIM_Base_Init(&htim11) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_TIM_PWM_Init(&htim11) != HAL_OK)
  {
    Error_Handler();
  }
  sConfigOC.OCMode = TIM_OCMODE_PWM1;
  sConfigOC.Pulse = 0;
  sConfigOC.OCPolarity = TIM_OCPOLARITY_HIGH;
  sConfigOC.OCFastMode = TIM_OCFAST_DISABLE;
  if (HAL_TIM_PWM_ConfigChannel(&htim11, &sConfigOC, TIM_CHANNEL_1) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN TIM11_Init 2 */

  /* USER CODE END TIM11_Init 2 */

}

/**
  * @brief TIM12 Initialization Function
  * @param None
  * @retval None
  */
static void MX_TIM12_Init(void)
{

  /* USER CODE BEGIN TIM12_Init 0 */

  /* USER CODE END TIM12_Init 0 */

  TIM_OC_InitTypeDef sConfigOC = {0};

  /* USER CODE BEGIN TIM12_Init 1 */

  /* USER CODE END TIM12_Init 1 */
  htim12.Instance = TIM12;
  htim12.Init.Prescaler = 83;
  htim12.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim12.Init.Period = 19999;
  htim12.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim12.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  if (HAL_TIM_PWM_Init(&htim12) != HAL_OK)
  {
    Error_Handler();
  }
  sConfigOC.OCMode = TIM_OCMODE_PWM1;
  sConfigOC.Pulse = 0;
  sConfigOC.OCPolarity = TIM_OCPOLARITY_HIGH;
  sConfigOC.OCFastMode = TIM_OCFAST_DISABLE;
  if (HAL_TIM_PWM_ConfigChannel(&htim12, &sConfigOC, TIM_CHANNEL_2) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN TIM12_Init 2 */

  /* USER CODE END TIM12_Init 2 */
  HAL_TIM_MspPostInit(&htim12);

}

/**
  * @brief USART2 Initialization Function
  * @param None
  * @retval None
  */
static void MX_USART2_UART_Init(void)
{

  /* USER CODE BEGIN USART2_Init 0 */

  /* USER CODE END USART2_Init 0 */

  /* USER CODE BEGIN USART2_Init 1 */

  /* USER CODE END USART2_Init 1 */
  huart2.Instance = USART2;
  huart2.Init.BaudRate = 9600;
  huart2.Init.WordLength = UART_WORDLENGTH_8B;
  huart2.Init.StopBits = UART_STOPBITS_1;
  huart2.Init.Parity = UART_PARITY_NONE;
  huart2.Init.Mode = UART_MODE_TX_RX;
  huart2.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart2.Init.OverSampling = UART_OVERSAMPLING_16;
  if (HAL_UART_Init(&huart2) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN USART2_Init 2 */

  /* USER CODE END USART2_Init 2 */

}

/**
  * @brief USART3 Initialization Function
  * @param None
  * @retval None
  */
static void MX_USART3_UART_Init(void)
{

  /* USER CODE BEGIN USART3_Init 0 */

  /* USER CODE END USART3_Init 0 */

  /* USER CODE BEGIN USART3_Init 1 */

  /* USER CODE END USART3_Init 1 */
  huart3.Instance = USART3;
  huart3.Init.BaudRate = 115200;
  huart3.Init.WordLength = UART_WORDLENGTH_8B;
  huart3.Init.StopBits = UART_STOPBITS_1;
  huart3.Init.Parity = UART_PARITY_NONE;
  huart3.Init.Mode = UART_MODE_TX_RX;
  huart3.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart3.Init.OverSampling = UART_OVERSAMPLING_16;
  if (HAL_UART_Init(&huart3) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN USART3_Init 2 */

  /* USER CODE END USART3_Init 2 */

}

/**
  * @brief GPIO Initialization Function
  * @param None
  * @retval None
  */
static void MX_GPIO_Init(void)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};
  /* USER CODE BEGIN MX_GPIO_Init_1 */

  /* USER CODE END MX_GPIO_Init_1 */

  /* GPIO Ports Clock Enable */
  __HAL_RCC_GPIOH_CLK_ENABLE();
  __HAL_RCC_GPIOA_CLK_ENABLE();
  __HAL_RCC_GPIOE_CLK_ENABLE();
  __HAL_RCC_GPIOB_CLK_ENABLE();
  __HAL_RCC_GPIOD_CLK_ENABLE();
  __HAL_RCC_GPIOC_CLK_ENABLE();

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(GPIOD, OLED4_Pin|OLED3_Pin|OLED2_Pin|OLED1_Pin, GPIO_PIN_RESET);

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(GPIOA, Buzzer_Pin|LED_Pin, GPIO_PIN_RESET);

  /*Configure GPIO pin : PB12 */
  GPIO_InitStruct.Pin = GPIO_PIN_12;
  GPIO_InitStruct.Mode = GPIO_MODE_IT_RISING;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);

  /*Configure GPIO pins : OLED4_Pin OLED3_Pin OLED2_Pin OLED1_Pin */
  GPIO_InitStruct.Pin = OLED4_Pin|OLED3_Pin|OLED2_Pin|OLED1_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOD, &GPIO_InitStruct);

  /*Configure GPIO pins : Buzzer_Pin LED_Pin */
  GPIO_InitStruct.Pin = Buzzer_Pin|LED_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

  /*Configure GPIO pins : USER_PB_Pin IMU_INT_Pin */
  GPIO_InitStruct.Pin = USER_PB_Pin|IMU_INT_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_IT_RISING;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(GPIOE, &GPIO_InitStruct);

  // --- Ultrasonic HCSR04 Pins ---
  // Trig -> PB14

  GPIO_InitStruct.Pin = GPIO_PIN_14;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);

  // Echo -> PC9
  GPIO_InitStruct.Pin = GPIO_PIN_9;
  GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(GPIOC, &GPIO_InitStruct);

  /* EXTI interrupt init*/
  HAL_NVIC_SetPriority(EXTI0_IRQn, 0, 0);
  HAL_NVIC_EnableIRQ(EXTI0_IRQn);

  HAL_NVIC_SetPriority(EXTI1_IRQn, 0, 0);
  HAL_NVIC_EnableIRQ(EXTI1_IRQn);

  /* USER CODE BEGIN MX_GPIO_Init_2 */
//  GPIO_InitStruct.Pin = GPIO_PIN_15;
//  GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
//  GPIO_InitStruct.Pull = GPIO_NOPULL;
//  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
//  GPIO_InitStruct.Alternate = GPIO_AF9_TIM12;
//  HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);
  /* USER CODE END MX_GPIO_Init_2 */
}

/* USER CODE BEGIN 4 */
/*
void OLED_show(void *argument, int y, int x) // display message on OLED panel
{
  //uint8_t hello[20]="Hello World";
  OLED_Init();
  OLED_Display_On();
//	OLED_ShowString(10,10,argument);
  OLED_ShowString(y, x, argument);
  OLED_Refresh_Gram();
}
*/

/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}
#ifdef USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
