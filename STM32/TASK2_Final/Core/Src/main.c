#include "main.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include "oled.h"
#include <stdarg.h>
#define PI 3.14159265f
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
ADC_HandleTypeDef hadc2;
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
void process_command(char *cmd);   // <-- add this prototype
static void MX_ADC1_Init(void);
static void MX_ADC2_Init(void);       // NEW
static void IR_Left_Read(void);       // NEW
static void IR_Right_Read(void);      // NEW

void     HCSR04_InitDWT(void);
uint32_t HCSR04_Read(void);
/* USART3 helpers + forward/stop/ask/turn (default RIGHT) */





/* ---------------------------- Globals ----------------------------- */
/* IMU globals (matching your project) */
volatile float ax_g, ay_g, az_g, gx_dps, gy_dps, gz_dps;

#define FILTER_ALPHA2  0.25f          // IIR smoothing 0..1 (0.2–0.3 works well)

static int   iDistanceL = 0, iDistanceR = 0;   // raw ADC 0..4095
      // left LPF state
static int   filtered_irreading_int = 0;
static int filtered_irreading = 0;

static float distanceirr = 0.0f;               // left distance (float, cm)
static float distanceirr_r = 0.0f;
static int   distanceir = 0;                   // left distance (int, cm)
static int   distanceir_right = 0;             // right distance (int, cm)

static float g_straight_dev_deg = 0.0f;
static float g_turn_residual_deg = 0.0f;
#define TURN_RESIDUAL_MIN_DEG (-10.0f)
#define TURN_RESIDUAL_MAX_DEG (10.0f)

/* Motor/Motion control */
static int PWM_TRIM = -350;       // negative slows the left side to fix veer-right
static const int16_t pwmMax = 4000;
static const int16_t pwmMin = 500;

// Global coordinates (cm) and heading (degrees)
static float robot_x = 0.0f;
static float robot_y = 0.0f;
static float robot_heading_deg = 0.0f;  // 0° = straight forward, + left, - right

// Encoder calibration (3rd pass):
static float COUNTS_PER_CM_L = 74.1515f;
static float COUNTS_PER_CM_R = 74.1515f;

/* ===== Gyro-straight control state & tuning ===== */
#define GYRO_YAW_SIGN (+1)      // flip to -1 if your yaw is inverted
static float g_bias_dps = 0.0f; // learned at first run
static float g_yaw_deg  = 0.0f; // integrated yaw
static float g_i_term   = 0.0f; // integral on heading error
static float g_target_deg = 0.0f;
static uint8_t g_gyro_inited = 0;

/* tune: start conservative */
static const float GS_KP = 2.0f;      // pwm / deg
static const float GS_KI = 0.05f;     // pwm / (deg*s)
static const float GS_KD = 0.8f;      // pwm / (deg/s)  (from yaw-rate)
static const float GS_I_MAX = 200.0f; // anti-windup clamp (pwm units)
static const int   GS_CORR_MAX = 300; // max +/- differential correction (pwm)

/* helper */
static inline float wrap180f(float x){
  while (x > 180.0f) x -= 360.0f;
  while (x < -180.0f) x += 360.0f;
  return x;
}


static float clampf(float v, float lo, float hi);
static int clampi(int v, int lo, int hi);
static int pwm_from_speed_cmps(float speed_cmps);

static void set_left_motor(int pwm);
static void set_right_motor(int pwm);

static float Apply_Turn_Compensation(float requested_signed_deg, int backward);
static void Update_Turn_Residual(float requested_signed_deg, float actual_signed_deg);


static void Motor_forward_PID(int pwmVal);
static float Apply_Straight_Deviation_To_Turn(float requested_signed_deg, int backward);

/* lock current heading as the target (call before a new straight leg if desired) */
static void GyroStraight_LockHeading(void){
  g_target_deg = wrap180f(g_yaw_deg);
  g_i_term = 0.0f;
}

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
//github copilot created forward declarations
void test_task2_three_steps(void);
void test_task2_three_steps(void);

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

/* Robot Movement Functions */
static float Drive_Turn_Angle(float target_turn_deg, int backwards);
void Drive_Straight_ToCM(float target_cm);

/* Straight helpers */
static inline void reset_encoders(void);
static inline int32_t left_ticks(void);
static inline int32_t right_ticks(void);
static float cm_travelled(void);
static int pwm_for_distance(float cm_left, int base_pwm);
static int pwm_from_speed_cmps(float speed_cmps);



/* --------------------- Helpers / small drivers -------------------- */
static inline void Servo_WriteUS(uint16_t us) {
  if (us < 500)  us = 500;
  if (us > 2500) us = 2500;
  __HAL_TIM_SET_COMPARE(&htim12, TIM_CHANNEL_2, us);
}
//static uint16_t SERVO_US_CENTER = 1500;  // nudge by ±10..20 if wheels not straight
//static uint16_t Steering_ToUS_original(int16_t steer_angle) {
//  if (steer_angle < -45) steer_angle = -45;
//  if (steer_angle >  45) steer_angle =  45;
//  int32_t us = SERVO_US_CENTER + (int32_t)steer_angle * ((2400 - 500) / 90);
//  __HAL_TIM_SET_COMPARE(&htim12, TIM_CHANNEL_2, (uint16_t)us);
//  return (uint16_t)us;
//}
// ===== SERVO CONFIG 2026GRP14=====

// Adjust this so wheels are perfectly straight at steer_angle = 0
#define STEER_CENTER 0
static uint16_t SERVO_US_CENTER = 1500;
static uint16_t SERVO_US_LEFT_MAX = 1900;
static uint16_t SERVO_US_RIGHT_MAX = 1900;

// These MUST be tuned so wheels never hit the chassis
#define SERVO_US_MIN   1100
#define SERVO_US_MAX   2500

// Maximum logical steering angle allowed by your chassis
#define STEER_ANGLE_LIMIT  40     // degrees
#define STEER_ANGLE_LIMIT_RIGHT 45
// -------- 2026 GRP14 servomotor controls ------------

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
         us = SERVO_US_CENTER + (int32_t)steer_angle * ((4000-1500) / 40);//this speific calculation works, add to new steering function
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



static inline void UART3_ACK(const char *fmt, ...) {
  char line[64];
  va_list ap; va_start(ap, fmt);
  int n = snprintf(line, sizeof(line), "ACK:");
  n += vsnprintf(line + n, sizeof(line) - n, fmt, ap);
  va_end(ap);
  if (n < (int)sizeof(line) - 2) { line[n++] = '\r'; line[n++] = '\n'; }
  HAL_UART_Transmit(&huart3, (uint8_t*)line, n, HAL_MAX_DELAY);
}


/* ====================== IR Sensors (ADC) ====================== */
/* Left on ADC1, Right on ADC2. If you want both on ADC1, say so and we’ll switch. */

static void IR_Left_Read(void)
{
    HAL_ADC_Start(&hadc1);
    HAL_ADC_PollForConversion(&hadc1, 10);
    iDistanceL = HAL_ADC_GetValue(&hadc1);
    HAL_ADC_Stop(&hadc1);

    // Low-pass filter
    static float filtered_irreading = 0.0f;
    filtered_irreading = FILTER_ALPHA2 * (float)iDistanceL
                       + (1.0f - FILTER_ALPHA2) * filtered_irreading;
    filtered_irreading_int = (int)filtered_irreading;

    // Distance mapping (power law) — tune constants to your calibration
    distanceirr = 92750.0f * powf((float)filtered_irreading_int, -1.15f);
    if (distanceirr < 0.0f) distanceirr = 0.0f;
    if (distanceirr > 999.0f) distanceirr = 999.0f;
    distanceir  = (int)distanceirr;  // cm (rounded down)
}
static float IR_Left_Read_try(void)// working IR read as of 17/03/2026
{
    HAL_ADC_Start(&hadc1);
    HAL_ADC_PollForConversion(&hadc1, 10);
    iDistanceL = HAL_ADC_GetValue(&hadc1);
    HAL_ADC_Stop(&hadc1);

    // Low-pass filter
    filtered_irreading = FILTER_ALPHA2 * (float)iDistanceL
                       + (1.0f - FILTER_ALPHA2) * filtered_irreading;
    filtered_irreading_int = (int)filtered_irreading;

    // Distance mapping (power law) — tune constants to your calibration
    distanceirr = 92750.0f * powf((float)filtered_irreading_int, -1.15f);
    if (distanceirr < 0.0f) distanceirr = 0.0f;
    if (distanceirr > 999.0f) distanceirr = 999.0f;
    distanceir  = (int)distanceirr;  // cm (rounded down)

    return distanceir;
}

static float IR_Right_Read_try(void)// working IR read as of 17/03/2026
{
    HAL_ADC_Start(&hadc2);
    HAL_ADC_PollForConversion(&hadc2, 10);
    iDistanceL = HAL_ADC_GetValue(&hadc2);
    HAL_ADC_Stop(&hadc2);

    // Low-pass filter
    filtered_irreading = FILTER_ALPHA2 * (float)iDistanceL
                       + (1.0f - FILTER_ALPHA2) * filtered_irreading;
    filtered_irreading_int = (int)filtered_irreading;

    // Distance mapping (power law) — tune constants to your calibration
    distanceirr_r = 92750.0f * powf((float)filtered_irreading_int, -1.15f);
    if (distanceirr_r < 0.0f) distanceirr_r = 0.0f;
    if (distanceirr_r > 999.0f) distanceirr_r = 999.0f;
    distanceir_right  = (int)distanceirr_r;  // cm (rounded down)

    return distanceir_right;
}

static void IR_Right_Read(void)
{
    HAL_ADC_Start(&hadc2);
    HAL_ADC_PollForConversion(&hadc2, 10);
    iDistanceR = HAL_ADC_GetValue(&hadc2);
    HAL_ADC_Stop(&hadc2);

    static float filtered_irreading_right = 0.0f;
    filtered_irreading_right = FILTER_ALPHA2 * (float)iDistanceR
                             + (1.0f - FILTER_ALPHA2) * filtered_irreading_right;
    int filtered_irreading_right_int = (int)filtered_irreading_right;

    float distanceirr_r = 92750.0f * powf((float)filtered_irreading_right_int, -1.15f);
    if (distanceirr_r < 0.0f) distanceirr_r = 0.0f;
    if (distanceirr_r > 999.0f) distanceirr_r = 999.0f;
    distanceir_right = (int)distanceirr_r;  // cm
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
static void Motor_forward(int pwmVal)
{
  static uint32_t last_ms = 0;
  static int did_bias = 0;

  uint32_t now = HAL_GetTick();
  float dt = (now - last_ms) * 0.001f;
  if (dt <= 0.0f || dt > 0.1f) dt = 0.01f; // guard if first call or long gap
  last_ms = now;

  /* 1) Read gyro and make gz_dps fresh here */
  int16_t ax, ay, az, gx, gy, gz;
  ICM20948_ReadRaw(&ax, &ay, &az, &gx, &gy, &gz);

  float gz_dps_local = ((float)gz / 131.0f) * (float)GYRO_YAW_SIGN; // ±250 dps FS assumed

  /* 2) One-time bias learn (robot should be still on first few calls) */
  if (!did_bias) {
    static int n = 0; static float acc = 0.0f;
    acc += gz_dps_local;
    if (++n >= 500) {                // ~0.5 s if called at ~1 kHz; here it’s called slower so it’ll take a bit longer
      g_bias_dps = acc / (float)n;
      did_bias = 1;
      g_gyro_inited = 1;
      /* lock present heading as target */
      g_yaw_deg = 0.0f;
      GyroStraight_LockHeading();
    }
  }

  /* 3) Integrate yaw (remove bias) */
  float yawrate = gz_dps_local - g_bias_dps; // deg/s
  g_yaw_deg = wrap180f(g_yaw_deg + yawrate * dt);

  /* 4) Heading error */
  float err = wrap180f(g_target_deg - g_yaw_deg);

  /* 5) PI + D (D from yaw rate for damping) */
  g_i_term += err * dt;
  if (g_i_term >  GS_I_MAX) g_i_term =  GS_I_MAX;
  if (g_i_term < -GS_I_MAX) g_i_term = -GS_I_MAX;

  float corr_f = GS_KP*err + GS_KI*g_i_term - GS_KD*yawrate;
  int corr = (int)lrintf(corr_f);
  if (corr >  GS_CORR_MAX) corr =  GS_CORR_MAX;
  if (corr < -GS_CORR_MAX) corr = -GS_CORR_MAX;

  /* 6) Compose and clamp */
  int left_pwm  = pwmVal + PWM_TRIM + corr;
  int right_pwm = pwmVal - PWM_TRIM - corr;

  if (left_pwm  > pwmMax)  left_pwm  = pwmMax;
  if (right_pwm > pwmMax)  right_pwm = pwmMax;
  if (left_pwm  < pwmMin)  left_pwm  = pwmMin;
  if (right_pwm < pwmMin)  right_pwm = pwmMin;

  /* 7) Drive forward */
  __HAL_TIM_SetCompare(&htim4, TIM_CHANNEL_3, left_pwm);
  __HAL_TIM_SetCompare(&htim4, TIM_CHANNEL_4, 0);
  __HAL_TIM_SetCompare(&htim1, TIM_CHANNEL_3, 0);
  __HAL_TIM_SetCompare(&htim1, TIM_CHANNEL_4, right_pwm);
}


static void Motor_forward_raw(int pwmVal) {
  int left_pwm  = pwmVal + PWM_TRIM;
  int right_pwm = pwmVal - PWM_TRIM;
  if (left_pwm  < pwmMin) left_pwm  = pwmMin;
  if (right_pwm < pwmMin) right_pwm = pwmMin;
  if (left_pwm  > pwmMax) left_pwm  = pwmMax;
  if (right_pwm > pwmMax) right_pwm = pwmMax;

  __HAL_TIM_SetCompare(&htim4, TIM_CHANNEL_3, left_pwm);
  __HAL_TIM_SetCompare(&htim4, TIM_CHANNEL_4, 0);
  __HAL_TIM_SetCompare(&htim1, TIM_CHANNEL_3, 0);
  __HAL_TIM_SetCompare(&htim1, TIM_CHANNEL_4, right_pwm);
}


/* Optional reverse */
static void Motor_reverse(int pwmVal) {
  __HAL_TIM_SetCompare(&htim4, TIM_CHANNEL_3, 0);
  __HAL_TIM_SetCompare(&htim4, TIM_CHANNEL_4, pwmVal);
  __HAL_TIM_SetCompare(&htim1, TIM_CHANNEL_3, pwmVal);
  __HAL_TIM_SetCompare(&htim1, TIM_CHANNEL_4, 0);
}

/* ================== HC-SR04 (PB14 TRIG, PC9 ECHO) ================== */
/* Uses DWT cycle counter for precise timing at SystemCoreClock (168 MHz) */

void HCSR04_InitDWT(void)
{
  /* Enable DWT cycle counter (Cortex-M4) */
  CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
  DWT->CYCCNT = 0;
  DWT->CTRL  |= DWT_CTRL_CYCCNTENA_Msk;
}

/* Fire a 10us TRIG, measure ECHO high time, convert to cm.
   Returns 0 on timeout or bad read. */
uint32_t HCSR04_Read(void)
{
  /* 10us trigger pulse on PB14 */
  HAL_GPIO_WritePin(GPIOB, GPIO_PIN_14, GPIO_PIN_RESET);
  for (volatile int i = 0; i < 50;  i++);   // short settle
  HAL_GPIO_WritePin(GPIOB, GPIO_PIN_14, GPIO_PIN_SET);
  for (volatile int i = 0; i < 300; i++);   // ~10 µs at 168 MHz
  HAL_GPIO_WritePin(GPIOB, GPIO_PIN_14, GPIO_PIN_RESET);

  /* Wait for echo to go HIGH (timeout ~50 ms) */
  uint32_t t_expire = HAL_GetTick() + 50;
  while (HAL_GPIO_ReadPin(GPIOC, GPIO_PIN_9) == GPIO_PIN_RESET) {
    if (HAL_GetTick() > t_expire) return 0;
  }

  /* Measure HIGH pulse width using DWT->CYCCNT */
  uint32_t start = DWT->CYCCNT;
  t_expire = HAL_GetTick() + 50;
  while (HAL_GPIO_ReadPin(GPIOC, GPIO_PIN_9) == GPIO_PIN_SET) {
    if (HAL_GetTick() > t_expire) return 0;
  }
  uint32_t stop = DWT->CYCCNT;

  /* Convert cycles -> microseconds */
  uint32_t cycles  = stop - start;
  uint32_t us      = cycles / (SystemCoreClock / 1000000U);

  /* Distance (cm) = (us * 34300 cm/s) / 2 / 1e6  ->  us * 343 / 20000 */
  return (uint32_t)((us * 343U) / 20000U);
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


void Drive_Straight_ToCM(float target_cm) // main forward and backwards code
{
    // -----------------------------
    // 1. Initialization
    // -----------------------------
    //stm tick is 1ms
    reset_encoders();
    Steering_ToUS(30);
    HAL_Delay(500);
    Steering_ToUS(0);
    HAL_Delay(500);  // center steering

    Gyro_ResetHeading();
    HAL_Delay(200);
    Gyro_ResetHeading();

    float prev_cm = 0.0f;
    float prev_left_enc = 0.0f;
    float prev_right_enc = 0.0f;
    float v_ref_cm_s = 0.0f;

    const float STOP_TOL_CM = 2.0f;
    const uint32_t OBSTACLE_STOP_CM = 35;   // <-- emergency stop distance (cm)
    const float VMAX_CM_S = 80.0f; //80cm/s max (5000 pwm)
    const float ACC_CM_S2 = 40.0f;

    const float FWD_KP = 0.18f;
    const float FWD_KI = 0.03f;
    const float FWD_KD = 0.0f;
    const float FWD_I_CLAMP = 4000.0f;
    const float FWD_CORR_CLAMP = 1200.0f;

    const int dir = (target_cm >= 0.0f) ? +1 : -1;
    target_cm = fabsf(target_cm);


    float a_integral = 0.0f;
    float l_v_integral = 0.0f;
    float r_v_integral = 0.0f;
    float last_a_error = 0.0f;
    float last_l_v_error = 0.0f;
    float last_r_v_error = 0.0f;
    uint32_t last_time = HAL_GetTick();
    uint32_t last_oled = HAL_GetTick();
    char oled_line[32];

    // -----------------------------
    // 2. Main Loop
    // -----------------------------
    while (1)
    {
    	if(target_cm > 0){
			// ----- Emergency stop for obstacles -----
			uint32_t distance_cm = HCSR04_Read();
			if (OBSTACLE_STOP_CM - 2 < distance_cm <= OBSTACLE_STOP_CM)
			{
				Motor_stop();
				OLED_Clear();
				OLED_ShowString(0, 0, (uint8_t*)"EMERGENCY STOP");
				snprintf(oled_line, sizeof(oled_line), "Dist:%lu cm", (unsigned long)distance_cm);
				OLED_ShowString(0, 16, (uint8_t*)oled_line);
				OLED_ShowString(0, 32, (uint8_t*)"Obstacle detected!");
				OLED_Refresh_Gram();
				break;
			}else if(distance_cm <=OBSTACLE_STOP_CM - 2){
				Drive_Straight_ToCM((float)distance_cm - OBSTACLE_STOP_CM);
				break;
			}
    	}


        // ----- Time Step -----
        uint32_t now = HAL_GetTick();
        float dt = (now - last_time) / 1000.0f;
        if (dt <= 0) dt = 0.001f;
        last_time = now;


        // ---- update gyro ----
        Gyro_UpdateFromIMU(dt);
        float heading = Gyro_GetHeadingDeg();   // YOUR convention: +left, -right

        // ----- Distance from Encoders -----
        float left_enc = left_ticks_signed();
        float left_cm  = left_enc / COUNTS_PER_CM_L;
        float right_enc = right_ticks_signed();
        float right_cm = right_enc / COUNTS_PER_CM_R;

        float delta_left_enc = left_enc - prev_left_enc;
        float delta_right_enc = right_enc - prev_right_enc;
        float meas_l_enc_s = delta_left_enc / dt;
        float meas_r_enc_s = delta_right_enc / dt;
        prev_left_enc = left_enc;
        prev_right_enc = right_enc;

        float avg_cm = (fabsf(left_cm) + fabsf(right_cm)) * 0.5f;
        float v_meas_cm_s = (avg_cm - prev_cm) / dt;


        // -----------------------------
        // 3. Speed Profile (Soft Braking)
        // -----------------------------
        float S = target_cm - avg_cm; //remaining distance
        float Ssv = S - (v_ref_cm_s * dt);
        float Sbr = (v_ref_cm_s * v_ref_cm_s) / (2.0f * ACC_CM_S2);
        float v_next = v_ref_cm_s + ACC_CM_S2 * dt;
        float Siv = S - (v_next * dt);
        float Sbriv = (v_next * v_next) / (2.0f * ACC_CM_S2);

        if (S < 0.0f || Ssv < Sbr) {
          v_ref_cm_s -= ACC_CM_S2 * dt;
        } else if (Siv > Sbriv && v_ref_cm_s < VMAX_CM_S) {
          v_ref_cm_s += ACC_CM_S2 * dt;
        }
        v_ref_cm_s = clampf(v_ref_cm_s, 0.0f, VMAX_CM_S);

        prev_cm = avg_cm;

        // -----------------------------
        // 4. Forward speed PID (enc/s)
        // -----------------------------
        float target_l_enc_s = (float)dir * v_ref_cm_s * COUNTS_PER_CM_L;
        float target_r_enc_s = (float)dir * v_ref_cm_s * COUNTS_PER_CM_R;

        float l_v_err = target_l_enc_s - meas_l_enc_s;
        l_v_integral += l_v_err * dt;
        l_v_integral = clampf(l_v_integral, -FWD_I_CLAMP, FWD_I_CLAMP);
        float l_v_derivative = (l_v_err - last_l_v_error) / dt;
        last_l_v_error = l_v_err;
        float corr_l = (FWD_KP * l_v_err) + (FWD_KI * l_v_integral) + (FWD_KD * l_v_derivative);
        corr_l = clampf(corr_l, -FWD_CORR_CLAMP, FWD_CORR_CLAMP);

        float r_v_err = target_r_enc_s - meas_r_enc_s;
        r_v_integral += r_v_err * dt;
        r_v_integral = clampf(r_v_integral, -FWD_I_CLAMP, FWD_I_CLAMP);
        float r_v_derivative = (r_v_err - last_r_v_error) / dt;
        last_r_v_error = r_v_err;
        float corr_r = (FWD_KP * r_v_err) + (FWD_KI * r_v_integral) + (FWD_KD * r_v_derivative);
        corr_r = clampf(corr_r, -FWD_CORR_CLAMP, FWD_CORR_CLAMP);

        int pwm_ff = pwm_from_speed_cmps(v_ref_cm_s);
        int pwm_l = clampi((int)lroundf((float)pwm_ff + corr_l), 0, pwmMax);
        int pwm_r = clampi((int)lroundf((float)pwm_ff + corr_r), 0, pwmMax);

        // -----------------------------
        // 4. Gyro PID
        // -----------------------------

        heading += gz_dps * dt;   // integrate gyro
        float a_error = -heading;   // target heading = 0
        a_integral += a_error * dt;
        float a_differential = (a_error - last_a_error) / dt;
        last_a_error = a_error;

        int heading_correction =
            (int)(/*HEADING_KP*/ 1500 * a_error +
                  /*HEADING_KI*/ 100 * a_integral +
                  /*HEADING_KD*/ 0 * a_differential);

        if (now - last_oled >= 100u)
        {
            last_oled = now;

            OLED_Clear();

            snprintf(oled_line, sizeof(oled_line), "D:%.2f R:%.2f", avg_cm, target_cm - avg_cm);
            OLED_ShowString(0, 0, (uint8_t*)oled_line);

            snprintf(oled_line, sizeof(oled_line), "Vm:%.2f Vt:%.2f", v_meas_cm_s, v_ref_cm_s);
            OLED_ShowString(0, 12, (uint8_t*)oled_line);

            snprintf(oled_line, sizeof(oled_line), "Head:%.2f", heading);
            OLED_ShowString(0, 24, (uint8_t*)oled_line);

            snprintf(oled_line, sizeof(oled_line), "Aerr:%.2f", a_error);
            OLED_ShowString(0, 36, (uint8_t*)oled_line);

            snprintf(oled_line, sizeof(oled_line), "Hcor:%d", heading_correction);
            OLED_ShowString(0, 48, (uint8_t*)oled_line);

            OLED_Refresh_Gram();
        }

        // -----------------------------
        // 5. Motor Mixing (Direction-Aware Trim)
        // -----------------------------
        int left_cmd  = dir * pwm_l - heading_correction;
        int right_cmd = dir * pwm_r + heading_correction;

        // Saturate PWM
        if (left_cmd  > pwmMax)  left_cmd  = pwmMax;
        if (left_cmd  < -pwmMax) left_cmd  = -pwmMax;
        if (right_cmd > pwmMax)  right_cmd = pwmMax;
        if (right_cmd < -pwmMax) right_cmd = -pwmMax;

        set_left_motor(left_cmd);
        set_right_motor(right_cmd);

        HAL_Delay(10);

        if (target_cm - avg_cm <= STOP_TOL_CM){
          break;
        }

    }

    // -----------------------------
    // 6. Stop Motors (always stop when exiting loop)
    // -----------------------------
    Motor_stop();
}
void Odometry_Update(float dL, float dR, float gyro_heading_deg)
{
    float d_center = (dL + dR) * 0.5f;

    // 🔥 ALWAYS trust gyro for heading
    float heading_rad = -gyro_heading_deg * PI / 180.0f;

    // Midpoint integration (simple & robust)
    robot_x += d_center * sinf(heading_rad);
    robot_y += d_center * cosf(heading_rad);

    // Directly set heading (NO accumulation)
    robot_heading_deg = -gyro_heading_deg;

    // Normalize
    if (robot_heading_deg > 180.0f)  robot_heading_deg -= 360.0f;
    if (robot_heading_deg < -180.0f) robot_heading_deg += 360.0f;
}

float Drive_Straight_ToCM_try(float target_cm, float OBSTACLE_STOP_CM_input)
{
    // -----------------------------
    // 1. Initialization
    // -----------------------------
    reset_encoders();
    Steering_ToUS(1);
    HAL_Delay(200);  // center steering

    Gyro_ResetHeading();
    HAL_Delay(200);
    Gyro_ResetHeading();

    float prev_cm_abs = 0.0f;
    float prev_left_enc = 0.0f;
    float prev_right_enc = 0.0f;
    float v_ref_cm_s = 0.0f;

    const float STOP_TOL_CM = 2.0f;
    const uint32_t OBSTACLE_STOP_CM = OBSTACLE_STOP_CM_input;   // emergency stop distance (cm)
    const float VMAX_CM_S = 100.0f; // max speed cm/s
    const float ACC_CM_S2 = 100.0f;  // acceleration cm/s²

    const float FWD_KP = 0.18f;
    const float FWD_KI = 0.09f;
    const float FWD_KD = 0.0f;
    const float FWD_I_CLAMP = 2000.0f;
    const float FWD_CORR_CLAMP = 1200.0f;

    const int dir = (target_cm >= 0.0f) ? +1 : -1;
    target_cm = fabsf(target_cm);

    float a_integral = 0.0f;
    float l_v_integral = 0.0f;
    float r_v_integral = 0.0f;
    float last_a_error = 0.0f;
    float last_l_v_error = 0.0f;
    float last_r_v_error = 0.0f;
    float avg_cm_signed = 0.0f;
    float avg_cm_abs = 0.0f;
    uint32_t last_time = HAL_GetTick();
    uint32_t last_oled = HAL_GetTick();
    char oled_line[32];

    // -----------------------------
    // 2. Main Loop
    // -----------------------------
    while (1)
    {
        // ----- Emergency stop for obstacles (forward only) -----
        if (dir > 0)
        {
            uint32_t distance_cm = HCSR04_Read();
            if ((distance_cm > 0u) && (distance_cm <= OBSTACLE_STOP_CM))
            {
                Motor_stop();
                OLED_Clear();
                OLED_ShowString(0, 0, (uint8_t*)"EMERGENCY STOP");
                snprintf(oled_line, sizeof(oled_line), "Dist:%lu cm", (unsigned long)distance_cm);
                OLED_ShowString(0, 16, (uint8_t*)oled_line);
                OLED_ShowString(0, 32, (uint8_t*)"Obstacle detected!");
                OLED_Refresh_Gram();
                break;
            }
        }

        // ----- Time Step -----
        uint32_t now = HAL_GetTick();
        float dt = (now - last_time) / 1000.0f;
        if(dt <= 0) dt = 0.001f;
        last_time = now;

        // ----- Update gyro -----
        Gyro_UpdateFromIMU(dt);
        float heading = Gyro_GetHeadingDeg(); // +left, -right

        // ----- Encoder distances -----
        float left_enc = left_ticks_signed();
        float left_cm  = left_enc / COUNTS_PER_CM_L;
        float right_enc = right_ticks_signed();
        float right_cm = right_enc / COUNTS_PER_CM_R;

        float delta_left_enc = left_enc - prev_left_enc;
        float delta_right_enc = right_enc - prev_right_enc;

        float dL = delta_left_enc / COUNTS_PER_CM_L;
        float dR = delta_right_enc / COUNTS_PER_CM_R;

        // 🔥 CRITICAL LINE (ADD THIS)
        Odometry_Update(dL, dR, Gyro_GetHeadingDeg());

        float meas_l_enc_s = delta_left_enc / dt;
        float meas_r_enc_s = delta_right_enc / dt;
        prev_left_enc = left_enc;
        prev_right_enc = right_enc;

        avg_cm_signed = (left_cm + right_cm) * 0.5f;
        avg_cm_abs = (fabsf(left_cm) + fabsf(right_cm)) * 0.5f;

        float v_meas_cm_s = (avg_cm_abs - prev_cm_abs) / dt;
        prev_cm_abs = avg_cm_abs;

        // ----- Speed Profile (soft acceleration/braking) -----
        float S = target_cm - avg_cm_abs;
        float v_next = v_ref_cm_s + ACC_CM_S2 * dt;
        float Sbriv = (v_next * v_next) / (2.0f * ACC_CM_S2);

        if(S < Sbriv || v_ref_cm_s > VMAX_CM_S)
            v_ref_cm_s -= ACC_CM_S2 * dt;
        else if(v_ref_cm_s < VMAX_CM_S)
            v_ref_cm_s += ACC_CM_S2 * dt;

        v_ref_cm_s = clampf(v_ref_cm_s, 0.0f, VMAX_CM_S);

        // ----- Forward speed PID -----
        float target_l_enc_s = (float)dir * v_ref_cm_s * COUNTS_PER_CM_L;
        float target_r_enc_s = (float)dir * v_ref_cm_s * COUNTS_PER_CM_R;

        float l_v_err = target_l_enc_s - meas_l_enc_s;
        l_v_integral += l_v_err * dt;
        l_v_integral = clampf(l_v_integral, -FWD_I_CLAMP, FWD_I_CLAMP);
        float l_v_derivative = (l_v_err - last_l_v_error) / dt;
        last_l_v_error = l_v_err;
        float corr_l = clampf(FWD_KP * l_v_err + FWD_KI * l_v_integral + FWD_KD * l_v_derivative,
                              -FWD_CORR_CLAMP, FWD_CORR_CLAMP);

        float r_v_err = target_r_enc_s - meas_r_enc_s;
        r_v_integral += r_v_err * dt;
        r_v_integral = clampf(r_v_integral, -FWD_I_CLAMP, FWD_I_CLAMP);
        float r_v_derivative = (r_v_err - last_r_v_error) / dt;
        last_r_v_error = r_v_err;
        float corr_r = clampf(FWD_KP * r_v_err + FWD_KI * r_v_integral + FWD_KD * r_v_derivative,
                              -FWD_CORR_CLAMP, FWD_CORR_CLAMP);

        int pwm_ff = pwm_from_speed_cmps(v_ref_cm_s);
        int pwm_l = clampi((int)lroundf((float)pwm_ff + corr_l), 0, pwmMax);
        int pwm_r = clampi((int)lroundf((float)pwm_ff + corr_r), 0, pwmMax);

        // ----- Gyro PID -----
        //heading += gz_dps * dt;
        float a_error = -heading;
        a_integral += a_error * dt;
        float a_differential = (a_error - last_a_error) / dt;
        last_a_error = a_error;
        //was 1500,10
        //good val for 60cm obs dist: 150
        int heading_correction = (int)(300 * a_error + 10 * a_integral + 0 * a_differential);

        // ----- OLED Debug -----
        if(now - last_oled >= 100)
        {
            last_oled = now;
            OLED_Clear();
            snprintf(oled_line, sizeof(oled_line), "D:%.2f Rem:%.2f", avg_cm_abs, target_cm - avg_cm_abs);
            OLED_ShowString(0, 0, (uint8_t*)oled_line);
            snprintf(oled_line, sizeof(oled_line), "Vm:%.2f Vt:%.2f", v_meas_cm_s, v_ref_cm_s);
            OLED_ShowString(0, 12, (uint8_t*)oled_line);
            snprintf(oled_line, sizeof(oled_line), "Head:%.2f", heading);
            OLED_ShowString(0, 24, (uint8_t*)oled_line);
            snprintf(oled_line, sizeof(oled_line), "Aerr:%.2f", a_error);
            OLED_ShowString(0, 36, (uint8_t*)oled_line);
            snprintf(oled_line, sizeof(oled_line), "Hcor:%d", heading_correction);
            OLED_ShowString(0, 48, (uint8_t*)oled_line);
            OLED_Refresh_Gram();
        }

        // ----- Motor Mixing -----
        int left_cmd  = dir * pwm_l - heading_correction;
        int right_cmd = dir * pwm_r + heading_correction;

        left_cmd  = clampi(left_cmd,  -pwmMax, pwmMax);
        right_cmd = clampi(right_cmd, -pwmMax, pwmMax);

        set_left_motor(left_cmd);
        set_right_motor(right_cmd);

        HAL_Delay(10);

        // ----- Stop condition -----
        if (target_cm - avg_cm_abs <= STOP_TOL_CM)
            break;
    }

    // -----------------------------
    // 3. Stop Motors
    // -----------------------------

    // ----- OLED Debug for global position -----
    char pos_buf[32];
    snprintf(pos_buf, sizeof(pos_buf), "X:%.1f Y:%.1f", robot_x, robot_y);
    OLED_ShowString(0, 48, (uint8_t*)pos_buf);  // choose line free on OLED

    snprintf(pos_buf, sizeof(pos_buf), "H:%.1f", robot_heading_deg);
    OLED_ShowString(0, 56, (uint8_t*)pos_buf);  // below the X/Y
    Motor_stop();
    return avg_cm_signed;  // signed distance travelled (+fwd, -rev)
}

float Drive_via_Ping(float target_cm) // drive using ping, uses ping for distance
{
	char pos_buf[32];
	OLED_Clear();
	  uint32_t distance_cm = HCSR04_Read();
	  float backwards_target_cm = target_cm - distance_cm;
	  snprintf(pos_buf, sizeof(pos_buf), "Distance_cm:%.1f", distance_cm);
	  OLED_ShowString(0, 0, (uint8_t*)pos_buf);  // choose line free on OLED

	  snprintf(pos_buf, sizeof(pos_buf), "Backwards_target_cm:%.1f", backwards_target_cm);
	  OLED_ShowString(0, 24, (uint8_t*)pos_buf);
	  OLED_Refresh_Gram();
	  HAL_Delay(200);
	  Drive_Straight_ToCM_try(-backwards_target_cm,35);

	  return -backwards_target_cm;
}
/* task 2 */
/* ---- USART3 one-line send/recv (CR/LF terminated) ---- */
void UART3_SendLine(const char *s)
{
  HAL_UART_Transmit(&huart3, (uint8_t*)s, strlen(s), HAL_MAX_DELAY);
  const char nl[2] = {'\n','\r'};
  HAL_UART_Transmit(&huart3, (uint8_t*)nl, 2, HAL_MAX_DELAY);
}

/* Read one line from USART3 until CR/LF or timeout. Returns true if non-empty. */
bool UART3_ReadLine(char *out, size_t max_len, uint32_t timeout_ms)
{
  uint32_t start = HAL_GetTick();
  size_t idx = 0;
  while ((HAL_GetTick() - start) < timeout_ms) {
    uint8_t ch;
    /* short per-byte wait to stay responsive */
    if (HAL_UART_Receive(&huart3, &ch, 1, 20) == HAL_OK) {
      if (ch == '\n' || ch == '\r') {
        if (idx > 0) break;      // end-of-line once we have data
        else continue;           // ignore leading CR/LF
      }
      if (idx + 1 < max_len) out[idx++] = (char)ch;
    }
  }
  out[idx] = '\0';
  return (idx > 0);
}

/* ---- Forward until obstacle on US, stop, ask RPi on USART3, execute its command (default RIGHT) ---- */
/* --- helper: forward until obstacle with small debounce + status OLED --- */
static void DriveForwardUntilObstacle(uint32_t stop_cm, int forward_pwm)
{
  const uint8_t N_CONSEC_NEAR = 3;
  uint8_t near_cnt = 0;

  Steering_ToUS(0);
  HAL_Delay(30);

  OLED_ShowString(0, 24, (uint8_t*)"Fwd until obstacle");
  OLED_Refresh_Gram();

  for (;;) {
    uint32_t d = HCSR04_Read();   // 0 on timeout
    if (d > 0 && d <= stop_cm) {
      if (++near_cnt >= N_CONSEC_NEAR) {
        Motor_stop();
        break;
      }
    } else {
      near_cnt = 0;
    }

    Motor_forward(forward_pwm);

    static uint32_t last = 0;
    uint32_t now = HAL_GetTick();
    if (now - last >= 120) {
      last = now;
      snprintf(buf, sizeof(buf), "US:%lu cm PWM:%d   ", (unsigned long)d, forward_pwm);
      OLED_ShowString(0, 36, (uint8_t*)buf);
      OLED_Refresh_Gram();
    }
    HAL_Delay(10);
  }

  Motor_stop();
  Steering_ToUS(0);
  HAL_Delay(80);
}

// Ask RPi for a TL-- / TR-- (or any command), with OLED status + fallback.
// Returns true if RPi replied before timeout. Out param always set.
static bool RPi_RequestTurnCommand(uint32_t timeout_ms, char *out_cmd, size_t out_sz)
{
    char reply[64] = {0};

    OLED_ShowString(0, 48, (uint8_t*)"RPi: CAPTURE...");
    OLED_Refresh_Gram();

    UART3_SendLine("CAPTURE");   // RPi should reply like TL-- or TR--

    bool got = UART3_ReadLine(reply, sizeof(reply), timeout_ms);

    if (got && reply[0]) {
        // show what we got
        snprintf(out_cmd, out_sz, "%s", reply);
        snprintf(buf, sizeof(buf), "RPi:%s      ", reply);
        OLED_ShowString(0, 56, (uint8_t*)buf);
        OLED_Refresh_Gram();
    } else {
        // fallback: default RIGHT
        snprintf(out_cmd, out_sz, "TR--");
        OLED_ShowString(0, 56, (uint8_t*)"RPi:TIMEOUT->TR--");
        OLED_Refresh_Gram();
    }
    return got;
}



void Drive_Forward_Until_LeftIR_Exceeds(int forward_pwm,
                                        int threshold_cm,
                                        int consec_needed,
                                        uint32_t poll_ms)
{

    if (forward_pwm < pwmMin) forward_pwm = pwmMin;
    if (forward_pwm > pwmMax) forward_pwm = pwmMax;
    if (threshold_cm < 1)     threshold_cm = 1;
    if (consec_needed < 1)    consec_needed = 1;
    if (poll_ms < 20)         poll_ms = 20;

    Steering_ToUS(0);
    HAL_Delay(50);

    Gyro_ResetHeading();   // important for straight driving

    uint8_t hit_cnt = 0;
    uint32_t last_poll = HAL_GetTick();

    OLED_ShowString(0,36,(uint8_t*)"Fwd until L>=TH");
    OLED_Refresh_Gram();

    while (1)
    {
        // ---- RUN MOTOR PID CONTINUOUSLY ----
    	Drive_Straight_ToCM(10);

        uint32_t now = HAL_GetTick();



            IR_Left_Read();
            IR_Right_Read();

            // Debug display
            char lbuf[32];
            snprintf(lbuf,sizeof(lbuf),"IR L:%3dcm R:%3dcm",distanceir,distanceir_right);
            OLED_ShowString(0,0,(uint8_t*)lbuf);

            char rbuf[32];
            snprintf(rbuf,sizeof(rbuf),"RawL:%4d RawR:%4d",iDistanceL,iDistanceR);
            OLED_ShowString(0,12,(uint8_t*)rbuf);

            OLED_Refresh_Gram();

            // ---- Debounce logic ----
            if (distanceir >= threshold_cm)
                hit_cnt++;
            else
                hit_cnt = 0;

            // ---- Stop condition ----
            if (hit_cnt >= consec_needed)
            {
                Motor_stop();

                OLED_ShowString(0,36,(uint8_t*)"Triggered L>=TH");
                OLED_Refresh_Gram();

                HAL_Delay(100);
                return;
            }
        }

        HAL_Delay(10);   // control loop ~100Hz

}

void Drive_Straight_ToCM_RightIR_Exceed(int dir) // James code but changed to ramp speed to a set PWM not by distance, but by fixed pwm
{
    // -----------------------------
    // 1. Initialization
    // -----------------------------
    //stm tick is 1ms
    reset_encoders();
    Steering_ToUS(30);
    HAL_Delay(500);
    Steering_ToUS(0);
    HAL_Delay(500);  // center steering

    Gyro_ResetHeading();
    HAL_Delay(200);
    Gyro_ResetHeading();

    float prev_cm = 0.0f;
    float prev_left_enc = 0.0f;
    float prev_right_enc = 0.0f;
    float v_ref_cm_s = 0.0f;

    const float STOP_TOL_CM = 2.0f;
    const uint32_t OBSTACLE_STOP_CM = 35;   // <-- emergency stop distance (cm)
    const float VMAX_CM_S = 80.0f; //80cm/s max (5000 pwm)
    const float ACC_CM_S2 = 40.0f;
    const float CRUISE_CM_S = 60.0f;   // fixed target speed(for pwm ramping by fixed pwm

    const float FWD_KP = 0.18f;
    const float FWD_KI = 0.03f;
    const float FWD_KD = 0.0f;
    const float FWD_I_CLAMP = 4000.0f;
    const float FWD_CORR_CLAMP = 1200.0f;




    float a_integral = 0.0f;
    float l_v_integral = 0.0f;
    float r_v_integral = 0.0f;
    float last_a_error = 0.0f;
    float last_l_v_error = 0.0f;
    float last_r_v_error = 0.0f;
    uint32_t last_time = HAL_GetTick();
    uint32_t last_oled = HAL_GetTick();
    char oled_line[32];

    // IR valaues
    float poll_ms = 40;
    float threshold_cm = 50;
    float consec_needed = 2;
    int hit_cnt = 0;

    // -----------------------------
    // 2. Main Loop
    // -----------------------------
    while (1)
    {

            IR_Right_Read();

            // Debug display
            char lbuf[32];
            snprintf(lbuf,sizeof(lbuf),"IR L:%3dcm R:%3dcm",distanceir,distanceir_right);
            OLED_ShowString(0,0,(uint8_t*)lbuf);

            char rbuf[32];
            snprintf(rbuf,sizeof(rbuf),"RawL:%4d RawR:%4d",iDistanceL,iDistanceR);
            OLED_ShowString(0,12,(uint8_t*)rbuf);

            OLED_Refresh_Gram();

            // ---- Debounce logic ----
            if (distanceir_right >= threshold_cm)// if distance IR detects anything is higher than threshold(basically if dont see black)
                hit_cnt++;
            else
                hit_cnt = 0;

            // ---- Stop condition ----
            if (hit_cnt >= consec_needed)
            {
                Motor_stop();

                OLED_ShowString(0,36,(uint8_t*)"Triggered R>=TH");
                OLED_Refresh_Gram();

                HAL_Delay(100);
                return;
            }




        // ----- Time Step -----
    // ----- Time Step -----


    	uint32_t now = HAL_GetTick();

        float dt = (now - last_time) / 1000.0f;
        if (dt <= 0) dt = 0.001f;
        last_time = now;


        // ---- update gyro ----
        Gyro_UpdateFromIMU(dt);
        float heading = Gyro_GetHeadingDeg();   // YOUR convention: +left, -right

        // ----- Distance from Encoders -----
        float left_enc = left_ticks_signed();
        float left_cm  = left_enc / COUNTS_PER_CM_L;
        float right_enc = right_ticks_signed();
        float right_cm = right_enc / COUNTS_PER_CM_R;

        float delta_left_enc = left_enc - prev_left_enc;
        float delta_right_enc = right_enc - prev_right_enc;
        float meas_l_enc_s = delta_left_enc / dt;
        float meas_r_enc_s = delta_right_enc / dt;
        prev_left_enc = left_enc;
        prev_right_enc = right_enc;

        float avg_cm = (fabsf(left_cm) + fabsf(right_cm)) * 0.5f;
        float v_meas_cm_s = (avg_cm - prev_cm) / dt;


//        // -----------------------------
//        // 3. Speed Profile (Soft Braking), distance-based
//        // -----------------------------
//        float S = target_cm - avg_cm; //remaining distance
//        float Ssv = S - (v_ref_cm_s * dt);
//        float Sbr = (v_ref_cm_s * v_ref_cm_s) / (2.0f * ACC_CM_S2);
//        float v_next = v_ref_cm_s + ACC_CM_S2 * dt;
//        float Siv = S - (v_next * dt);
//        float Sbriv = (v_next * v_next) / (2.0f * ACC_CM_S2);
//
//        if (S < 0.0f || Ssv < Sbr) {
//          v_ref_cm_s -= ACC_CM_S2 * dt;
//        } else if (Siv > Sbriv && v_ref_cm_s < VMAX_CM_S) {
//          v_ref_cm_s += ACC_CM_S2 * dt;
//        }
//        v_ref_cm_s = clampf(v_ref_cm_s, 0.0f, VMAX_CM_S);
//
//        prev_cm = avg_cm;

        // -----------------------------
		// 3. Fixed-speed ramp
		// -----------------------------

        float target_speed_cm_s = clampf(CRUISE_CM_S, 0.0f, VMAX_CM_S);

                if (v_ref_cm_s < target_speed_cm_s)
                {
                    v_ref_cm_s += ACC_CM_S2 * dt;
                    if (v_ref_cm_s > target_speed_cm_s)
                        v_ref_cm_s = target_speed_cm_s;
                }
                else if (v_ref_cm_s > target_speed_cm_s)
                {
                    v_ref_cm_s -= ACC_CM_S2 * dt;
                    if (v_ref_cm_s < target_speed_cm_s)
                        v_ref_cm_s = target_speed_cm_s;
                }

        // -----------------------------
        // 4. Forward speed PID (enc/s)
        // -----------------------------
        float target_l_enc_s = (float)dir * v_ref_cm_s * COUNTS_PER_CM_L;
        float target_r_enc_s = (float)dir * v_ref_cm_s * COUNTS_PER_CM_R;

        float l_v_err = target_l_enc_s - meas_l_enc_s;
        l_v_integral += l_v_err * dt;
        l_v_integral = clampf(l_v_integral, -FWD_I_CLAMP, FWD_I_CLAMP);
        float l_v_derivative = (l_v_err - last_l_v_error) / dt;
        last_l_v_error = l_v_err;
        float corr_l = (FWD_KP * l_v_err) + (FWD_KI * l_v_integral) + (FWD_KD * l_v_derivative);
        corr_l = clampf(corr_l, -FWD_CORR_CLAMP, FWD_CORR_CLAMP);

        float r_v_err = target_r_enc_s - meas_r_enc_s;
        r_v_integral += r_v_err * dt;
        r_v_integral = clampf(r_v_integral, -FWD_I_CLAMP, FWD_I_CLAMP);
        float r_v_derivative = (r_v_err - last_r_v_error) / dt;
        last_r_v_error = r_v_err;
        float corr_r = (FWD_KP * r_v_err) + (FWD_KI * r_v_integral) + (FWD_KD * r_v_derivative);
        corr_r = clampf(corr_r, -FWD_CORR_CLAMP, FWD_CORR_CLAMP);

        int pwm_ff = pwm_from_speed_cmps(v_ref_cm_s);
        int pwm_l = clampi((int)lroundf((float)pwm_ff + corr_l), 0, pwmMax);
        int pwm_r = clampi((int)lroundf((float)pwm_ff + corr_r), 0, pwmMax);

        // -----------------------------
        // 4. Gyro PID
        // -----------------------------

        heading += gz_dps * dt;   // integrate gyro
        float a_error = -heading;   // target heading = 0
        a_integral += a_error * dt;
        float a_differential = (a_error - last_a_error) / dt;
        last_a_error = a_error;

        int heading_correction =
            (int)(/*HEADING_KP*/ 1500 * a_error +
                  /*HEADING_KI*/ 100 * a_integral +
                  /*HEADING_KD*/ 0 * a_differential);

        if (now - last_oled >= 100u)
        {
            last_oled = now;

            OLED_Clear();



            snprintf(oled_line, sizeof(oled_line), "Vm:%.2f Vt:%.2f", v_meas_cm_s, v_ref_cm_s);
            OLED_ShowString(0, 12, (uint8_t*)oled_line);

            snprintf(oled_line, sizeof(oled_line), "Head:%.2f", heading);
            OLED_ShowString(0, 24, (uint8_t*)oled_line);

            snprintf(oled_line, sizeof(oled_line), "Aerr:%.2f", a_error);
            OLED_ShowString(0, 36, (uint8_t*)oled_line);

            snprintf(oled_line, sizeof(oled_line), "Hcor:%d", heading_correction);
            OLED_ShowString(0, 48, (uint8_t*)oled_line);

            OLED_Refresh_Gram();
        }

        // -----------------------------
        // 5. Motor Mixing (Direction-Aware Trim)
        // -----------------------------
        int left_cmd  = dir * pwm_l - heading_correction;
        int right_cmd = dir * pwm_r + heading_correction;

        // Saturate PWM
        if (left_cmd  > pwmMax)  left_cmd  = pwmMax;
        if (left_cmd  < -pwmMax) left_cmd  = -pwmMax;
        if (right_cmd > pwmMax)  right_cmd = pwmMax;
        if (right_cmd < -pwmMax) right_cmd = -pwmMax;

        set_left_motor(left_cmd);
        set_right_motor(right_cmd);

        HAL_Delay(10);
        }

    // -----------F=------------------
        // 6. Stop Motors (always stop when exiting loop)
        // -----------------------------
        Motor_stop();

    }

void Drive_Straight_ToCM_LeftIR_Exceed_old(int dir) // James code but changed to ramp speed to a set PWM not by distance, but by fixed pwm
{
    // -----------------------------
    // 1. Initialization
    // -----------------------------
    //stm tick is 1ms
    reset_encoders();
    Steering_ToUS(30);
    HAL_Delay(500);
    Steering_ToUS(0);
    HAL_Delay(500);  // center steering

    Gyro_ResetHeading();
    HAL_Delay(200);
    Gyro_ResetHeading();

    float prev_cm = 0.0f;
    float prev_left_enc = 0.0f;
    float prev_right_enc = 0.0f;
    float v_ref_cm_s = 0.0f;

    const float STOP_TOL_CM = 2.0f;
    const uint32_t OBSTACLE_STOP_CM = 35;   // <-- emergency stop distance (cm)
    const float VMAX_CM_S = 80.0f; //80cm/s max (5000 pwm)
    const float ACC_CM_S2 = 40.0f;
    const float CRUISE_CM_S = 60.0f;   // fixed target speed(for pwm ramping by fixed pwm

    const float FWD_KP = 0.18f;
    const float FWD_KI = 0.03f;
    const float FWD_KD = 0.0f;
    const float FWD_I_CLAMP = 4000.0f;
    const float FWD_CORR_CLAMP = 1200.0f;

    float a_integral = 0.0f;
    float l_v_integral = 0.0f;
    float r_v_integral = 0.0f;
    float last_a_error = 0.0f;
    float last_l_v_error = 0.0f;
    float last_r_v_error = 0.0f;
    uint32_t last_time = HAL_GetTick();
    uint32_t last_oled = HAL_GetTick();
    char oled_line[32];

    // IR values + left-wall hold settings
    uint32_t poll_ms = 40u;
    float threshold_cm = 100.0f;
    int consec_needed = 2;
    int hit_cnt = 0;

    const float WALL_TARGET_CM = 25.0f; //cm away from wall
    const float WALL_KP = 0.35f;
    const float WALL_KI = 0.04f;
    const float WALL_KD = 0.08f;
    const float WALL_I_CLAMP = 30.0f;
    const float WALL_HEADING_BIAS_MAX_DEG = 10.0f;
    const float WALL_VALID_MIN_CM = 10.0f;
    const float WALL_VALID_MAX_CM = 50.0f;

    float wall_i = 0.0f;
    float wall_prev_err = 0.0f;
    float wall_heading_bias_deg = 0.0f;
    float wall_err_cm = 0.0f;
    float left_ir_cm = 0.0f;
    uint8_t left_ir_valid = 0u;
    uint32_t last_ir_ms = HAL_GetTick();

    left_ir_cm = (float)IR_Left_Read_try();
    left_ir_valid = (left_ir_cm >= WALL_VALID_MIN_CM && left_ir_cm <= WALL_VALID_MAX_CM) ? 1u : 0u;

    // -----------------------------
    // 2. Main Loop
    // -----------------------------
    while (1)
    {

        uint32_t now = HAL_GetTick();

        // Read left IR sensor and update wall controller at polling cadence
        if (now - last_ir_ms >= poll_ms)
        {
            float ir_dt_s = (now - last_ir_ms) / 1000.0f;
            if (ir_dt_s <= 0.0f) ir_dt_s = 0.001f;
            last_ir_ms = now;

            left_ir_cm = (float)IR_Left_Read_try();

            // ---- Debounce logic for safety stop ----
            if (distanceir >= threshold_cm)
                hit_cnt++;
            else
                hit_cnt = 0;

            // ---- Stop condition ----
            if (hit_cnt >= consec_needed)
            {
                Motor_stop();
                OLED_ShowString(0,36,(uint8_t*)"Triggered L>=TH");
                OLED_Refresh_Gram();
                HAL_Delay(100);
                return;
            }

            left_ir_valid = (left_ir_cm >= WALL_VALID_MIN_CM && left_ir_cm <= WALL_VALID_MAX_CM) ? 1u : 0u;

            if ((dir > 0) && left_ir_valid)
            {
                float wall_d;
                wall_err_cm = left_ir_cm - WALL_TARGET_CM;
                wall_i += wall_err_cm * ir_dt_s;
                wall_i = clampf(wall_i, -WALL_I_CLAMP, WALL_I_CLAMP);
                wall_d = (wall_err_cm - wall_prev_err) / ir_dt_s;
                wall_prev_err = wall_err_cm;
wall_heading_bias_deg = (WALL_KP * wall_err_cm) +
                                        (WALL_KI * wall_i) +
                                        (WALL_KD * wall_d);
                wall_heading_bias_deg = clampf(wall_heading_bias_deg,
                                               -WALL_HEADING_BIAS_MAX_DEG,
                                                WALL_HEADING_BIAS_MAX_DEG);
            }
            else
            {
                wall_i = 0.0f;
                wall_prev_err = 0.0f;
                wall_err_cm = 0.0f;
                wall_heading_bias_deg = 0.0f;
            }
        }

        float dt = (now - last_time) / 1000.0f;
        if (dt <= 0) dt = 0.001f;
        last_time = now;


        // ---- update gyro ----
        Gyro_UpdateFromIMU(dt);
        float heading = Gyro_GetHeadingDeg();   // YOUR convention: +left, -right
        float heading_target_deg = ((dir > 0) && left_ir_valid) ? wall_heading_bias_deg : 0.0f;

        // ----- Distance from Encoders -----
        float left_enc = left_ticks_signed();
        float left_cm  = left_enc / COUNTS_PER_CM_L;
        float right_enc = right_ticks_signed();
        float right_cm = right_enc / COUNTS_PER_CM_R;

        float delta_left_enc = left_enc - prev_left_enc;
        float delta_right_enc = right_enc - prev_right_enc;
        float meas_l_enc_s = delta_left_enc / dt;
        float meas_r_enc_s = delta_right_enc / dt;
        prev_left_enc = left_enc;
        prev_right_enc = right_enc;

        float avg_cm = (fabsf(left_cm) + fabsf(right_cm)) * 0.5f;
        float v_meas_cm_s = (avg_cm - prev_cm) / dt;
        prev_cm = avg_cm;


//        // -----------------------------
//        // 3. Speed Profile (Soft Braking), distance-based
//        // -----------------------------
//        float S = target_cm - avg_cm; //remaining distance
//        float Ssv = S - (v_ref_cm_s * dt);
//        float Sbr = (v_ref_cm_s * v_ref_cm_s) / (2.0f * ACC_CM_S2);
//        float v_next = v_ref_cm_s + ACC_CM_S2 * dt;
//        float Siv = S - (v_next * dt);
//        float Sbriv = (v_next * v_next) / (2.0f * ACC_CM_S2);
//
//        if (S < 0.0f || Ssv < Sbr) {
//          v_ref_cm_s -= ACC_CM_S2 * dt;
//        } else if (Siv > Sbriv && v_ref_cm_s < VMAX_CM_S) {
//          v_ref_cm_s += ACC_CM_S2 * dt;
//        }
//        v_ref_cm_s = clampf(v_ref_cm_s, 0.0f, VMAX_CM_S);
//
//        prev_cm = avg_cm;

        // -----------------------------
    // 3. Fixed-speed ramp
    // -----------------------------

        float target_speed_cm_s = clampf(CRUISE_CM_S, 0.0f, VMAX_CM_S);

                if (v_ref_cm_s < target_speed_cm_s)
                {
                    v_ref_cm_s += ACC_CM_S2 * dt;
                    if (v_ref_cm_s > target_speed_cm_s)
                        v_ref_cm_s = target_speed_cm_s;
                }
                else if (v_ref_cm_s > target_speed_cm_s)
                {
                    v_ref_cm_s -= ACC_CM_S2 * dt;
                    if (v_ref_cm_s < target_speed_cm_s)
                        v_ref_cm_s = target_speed_cm_s;
                }

        // -----------------------------
        // 4. Forward speed PID (enc/s)
        // -----------------------------
        float target_l_enc_s = (float)dir * v_ref_cm_s * COUNTS_PER_CM_L;
        float target_r_enc_s = (float)dir * v_ref_cm_s * COUNTS_PER_CM_R;

        float l_v_err = target_l_enc_s - meas_l_enc_s;
        l_v_integral += l_v_err * dt;
        l_v_integral = clampf(l_v_integral, -FWD_I_CLAMP, FWD_I_CLAMP);
        float l_v_derivative = (l_v_err - last_l_v_error) / dt;
        last_l_v_error = l_v_err;
        float corr_l = (FWD_KP * l_v_err) + (FWD_KI * l_v_integral) + (FWD_KD * l_v_derivative);
        corr_l = clampf(corr_l, -FWD_CORR_CLAMP, FWD_CORR_CLAMP);
float r_v_err = target_r_enc_s - meas_r_enc_s;
        r_v_integral += r_v_err * dt;
        r_v_integral = clampf(r_v_integral, -FWD_I_CLAMP, FWD_I_CLAMP);
        float r_v_derivative = (r_v_err - last_r_v_error) / dt;
        last_r_v_error = r_v_err;
        float corr_r = (FWD_KP * r_v_err) + (FWD_KI * r_v_integral) + (FWD_KD * r_v_derivative);
        corr_r = clampf(corr_r, -FWD_CORR_CLAMP, FWD_CORR_CLAMP);

        int pwm_ff = pwm_from_speed_cmps(v_ref_cm_s);
        int pwm_l = clampi((int)lroundf((float)pwm_ff + corr_l), 0, pwmMax);
        int pwm_r = clampi((int)lroundf((float)pwm_ff + corr_r), 0, pwmMax);

        // -----------------------------
        // 4. Gyro PID
        // -----------------------------

        float a_error = heading_target_deg - heading;
        a_integral += a_error * dt;
        float a_differential = (a_error - last_a_error) / dt;
        last_a_error = a_error;

        int heading_correction =
            (int)(/*HEADING_KP*/ 1500 * a_error +
                  /*HEADING_KI*/ 100 * a_integral +
                  /*HEADING_KD*/ 0 * a_differential);

        if (now - last_oled >= 100u)
        {
            last_oled = now;

            OLED_Clear();



            snprintf(oled_line, sizeof(oled_line), "LIR:%3d E:%+.1f", (int)lroundf(left_ir_cm), wall_err_cm);
            OLED_ShowString(0, 0, (uint8_t*)oled_line);

            snprintf(oled_line, sizeof(oled_line), "Vm:%.2f Vt:%.2f", v_meas_cm_s, v_ref_cm_s);
            OLED_ShowString(0, 12, (uint8_t*)oled_line);

            snprintf(oled_line, sizeof(oled_line), "Hd:%.2f Tg:%.2f", heading, heading_target_deg);
            OLED_ShowString(0, 24, (uint8_t*)oled_line);

            snprintf(oled_line, sizeof(oled_line), "Aerr:%.2f", a_error);
            OLED_ShowString(0, 36, (uint8_t*)oled_line);

            snprintf(oled_line, sizeof(oled_line), "Hcor:%d", heading_correction);
            OLED_ShowString(0, 48, (uint8_t*)oled_line);

            OLED_Refresh_Gram();
        }

        // -----------------------------
        // 5. Motor Mixing (Direction-Aware Trim)
        // -----------------------------
        int left_cmd  = dir * pwm_l - heading_correction;
        int right_cmd = dir * pwm_r + heading_correction;

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

float Drive_Straight_ToCM_RightIR_Exceed2(int dir)//working code for right IR exceed as of 18/03/2026
{
    // -----------------------------
    // 1. Initialization
    // -----------------------------
    reset_encoders();
    Steering_ToUS(1);
    HAL_Delay(200);  // center steering

    Gyro_ResetHeading();
    HAL_Delay(200);
    Gyro_ResetHeading();

    static float robot_x = 0.0f;
    static float robot_y = 0.0f;
    static float robot_heading_deg = 0.0f;

    float delta_left_cm = 0.0f;
    float delta_right_cm = 0.0f;
    float d_center_cm = 0.0f;

    float avg_cm =0.0f;


    float prev_cm = 0.0f;
    float prev_left_enc = 0.0f;
    float prev_right_enc = 0.0f;
    float v_ref_cm_s = 0.0f;

    const float VMAX_CM_S = 100.0f;
    const float ACC_CM_S2 = 30.0f;
    const float CRUISE_CM_S = 30.0f;

    const float FWD_KP = 0.18f; //0.18
    const float FWD_KI = 0.02f; //0.03
    const float FWD_KD = 0.0f;
    const float FWD_I_CLAMP = 4000.0f;
    const float FWD_CORR_CLAMP = 1200.0f;

    const float WALL_KP = 8.0f; //500
    const float WALL_KI = 0.00f; //0.04
    const float WALL_KD = 1.00f; //0.08
    const float WALL_I_CLAMP = 300.0f;
    const float WALL_HEADING_BIAS_MAX_DEG = 1000.0f;
    const float WALL_VALID_MIN_CM = 10.0f;
    const float WALL_VALID_MAX_CM = 50.0f;

    float l_v_integral = 0.0f, r_v_integral = 0.0f, a_integral = 0.0f;
    float last_l_v_error = 0.0f, last_r_v_error = 0.0f, last_a_error = 0.0f;
    float wall_i = 0.0f, wall_prev_err = 0.0f;
    float wall_heading_bias_deg = 0.0f;

    int hit_cnt = 0;
    float right_ir_cm = 0.0f;
    uint8_t right_ir_valid = 0u;

    int prev_pwm_l = 0, prev_pwm_r = 0;

    uint32_t last_time = HAL_GetTick();
    uint32_t last_ir_ms = HAL_GetTick();
    uint32_t last_oled_ms = HAL_GetTick();

    char oled_line[32];

    float WALL_TARGET_CM = 21.0f;
    int ir_loop_count = 0;


    // -----------------------------
    // 2. Main Loop (Non-blocking)
    // -----------------------------
    while(1)
    {
        uint32_t now = HAL_GetTick();
        float dt = (now - last_time) / 1000.0f;
        if (dt <= 0.0f) dt = 0.001f;
        last_time = now;

        // -----------------------------
        // a) Read right IR every 40ms
        // -----------------------------
        if (now - last_ir_ms >= 40u)
        {
            float ir_dt_s = (now - last_ir_ms) / 1000.0f;
            if (ir_dt_s <= 0.0f) ir_dt_s = 0.001f;
            last_ir_ms = now;


            right_ir_cm = (float)IR_Right_Read_try();
            if (ir_loop_count == 8){
            	WALL_TARGET_CM = right_ir_cm;
            }
            right_ir_valid = (right_ir_cm >= WALL_VALID_MIN_CM && right_ir_cm <= WALL_VALID_MAX_CM) ? 1u : 0u;

            // ---- Safety stop
            if (right_ir_cm >= 75.0f) hit_cnt++;
            else hit_cnt = 0;

            if (hit_cnt >= 2)
            {
                Motor_stop();
                OLED_ShowString(0,36,(uint8_t*)"Triggered R>=TH");
                OLED_Refresh_Gram();
                return avg_cm;
            }

            // ---- Wall PID
            if ((dir > 0) && right_ir_valid && ir_loop_count > 10)
            {
                wall_prev_err = wall_prev_err;
                float wall_err = right_ir_cm - WALL_TARGET_CM;
                wall_i += wall_err * ir_dt_s;
                wall_i = clampf(wall_i, -WALL_I_CLAMP, WALL_I_CLAMP);
                float wall_d = (wall_err - wall_prev_err)/ir_dt_s;
                wall_prev_err = wall_err;

                wall_heading_bias_deg = WALL_KP*wall_err + WALL_KI*wall_i + WALL_KD*wall_d;
            }
            else
            {
                wall_i = 0.0f;
                wall_prev_err = 0.0f;
                wall_heading_bias_deg = 0.0f;
            }

            ir_loop_count++;
        }

        // -----------------------------
        // b) Read gyro
        // -----------------------------
        Gyro_UpdateFromIMU(dt);
        float heading = Gyro_GetHeadingDeg();
        float heading_target_deg = ((dir > 0) && right_ir_valid) ? wall_heading_bias_deg : 0.0f;

        // -----------------------------
        // c) Read encoders
        // -----------------------------

        float left_enc = left_ticks_signed();
        float right_enc = right_ticks_signed();
        float left_cm  = left_enc / COUNTS_PER_CM_L;
        float right_cm = right_enc / COUNTS_PER_CM_R;
        avg_cm = (fabsf(left_cm) + fabsf(right_cm))*0.5f;

        float delta_left_enc = left_enc - prev_left_enc;
        float delta_right_enc = right_enc - prev_right_enc;
        prev_left_enc = left_enc;
        prev_right_enc = right_enc;

        float meas_l_enc_s = delta_left_enc/dt;
        float meas_r_enc_s = delta_right_enc/dt;

        float v_meas_cm_s = (avg_cm - prev_cm)/dt;
        prev_cm = avg_cm;

        delta_left_cm = delta_left_enc / COUNTS_PER_CM_L;
        delta_right_cm = delta_right_enc / COUNTS_PER_CM_R;

        Odometry_Update(delta_left_cm, delta_right_cm, heading);
        // -----------------------------
        // d) Speed ramp
        // -----------------------------
        float target_speed_cm_s = clampf(CRUISE_CM_S, 0.0f, VMAX_CM_S);
        if (v_ref_cm_s < target_speed_cm_s) v_ref_cm_s += ACC_CM_S2*dt;
        else if (v_ref_cm_s > target_speed_cm_s) v_ref_cm_s -= ACC_CM_S2*dt;
        v_ref_cm_s = clampf(v_ref_cm_s, 0.0f, target_speed_cm_s);

        // -----------------------------
        // e) Forward speed PID
        // -----------------------------
        float target_l_enc_s = dir*v_ref_cm_s*COUNTS_PER_CM_L;
        float target_r_enc_s = dir*v_ref_cm_s*COUNTS_PER_CM_R;

        float l_err = target_l_enc_s - meas_l_enc_s;
        l_v_integral += l_err*dt;
        l_v_integral = clampf(l_v_integral, -FWD_I_CLAMP, FWD_I_CLAMP);
        float l_deriv = (l_err - last_l_v_error)/dt;
        last_l_v_error = l_err;
        float corr_l = FWD_KP*l_err + FWD_KI*l_v_integral + FWD_KD*l_deriv;
        corr_l = clampf(corr_l, -FWD_CORR_CLAMP, FWD_CORR_CLAMP);

        float r_err = target_r_enc_s - meas_r_enc_s;
        r_v_integral += r_err*dt;
        r_v_integral = clampf(r_v_integral, -FWD_I_CLAMP, FWD_I_CLAMP);
        float r_deriv = (r_err - last_r_v_error)/dt;
        last_r_v_error = r_err;
        float corr_r = FWD_KP*r_err + FWD_KI*r_v_integral + FWD_KD*r_deriv;
        corr_r = clampf(corr_r, -FWD_CORR_CLAMP, FWD_CORR_CLAMP);

        int pwm_ff = pwm_from_speed_cmps(v_ref_cm_s);
        int pwm_l = clampi((int)lroundf(pwm_ff + corr_l), 0, pwmMax);
        int pwm_r = clampi((int)lroundf(pwm_ff + corr_r), 0, pwmMax);

        // -----------------------------
        // f) Heading PID
        // -----------------------------
        float a_err = heading_target_deg - heading;
        a_integral += a_err*dt;
        float a_deriv = (a_err - last_a_error)/dt;
        last_a_error = a_err;

        int heading_correction = (int)(0*a_err + 0*a_integral);

        // -----------------------------
        // g) Motor mixing + smooth PWM
        // -----------------------------
        int left_cmd = dir*pwm_l - heading_correction + wall_heading_bias_deg;
        int right_cmd = dir*pwm_r + heading_correction - wall_heading_bias_deg;

        // PWM low-pass filter
        left_cmd = prev_pwm_l*0.7f + left_cmd*0.3f;
        right_cmd = prev_pwm_r*0.7f + right_cmd*0.3f;
        prev_pwm_l = left_cmd;
        prev_pwm_r = right_cmd;

        // Saturate
        left_cmd  = clampi(left_cmd, -pwmMax, pwmMax);
        right_cmd = clampi(right_cmd, -pwmMax, pwmMax);

        set_left_motor(left_cmd);
        set_right_motor(right_cmd);

        // -----------------------------
        // h) OLED update every 100ms
        // -----------------------------
        if (now - last_oled_ms >= 100u)
        {
            last_oled_ms = now;
            OLED_Clear();
            snprintf(oled_line, sizeof(oled_line), "T:%.1f", WALL_TARGET_CM);
            OLED_ShowString(0,0,(uint8_t*)oled_line);
            snprintf(oled_line, sizeof(oled_line), "L:%d R:%d", left_cmd, right_cmd);
            OLED_ShowString(0,12,(uint8_t*)oled_line);
//            rintf(oled_line, sizeof(oled_line), "Vm:%.2f Vt:%.2f", v_meas_cm_s, v_ref_cm_s);
//            OLED_ShowString(0,12,(uint8_t*)oled_line);
            snprintf(oled_line, sizeof(oled_line), "r_ir:%.2f",  right_ir_cm);
            OLED_ShowString(0,24,(uint8_t*)oled_line);
            snprintf(oled_line, sizeof(oled_line), "Average_cm:%.2f", avg_cm);
            OLED_ShowString(0,36,(uint8_t*)oled_line);
            OLED_Refresh_Gram();
        }
    }

    return avg_cm;
}

float Drive_Straight_ToCM_LeftIR_Exceed2(int dir)//working code for right IR exceed as of 18/03/2026
{

	    // -----------------------------
	    // 1. Initialization
	    // -----------------------------
	    reset_encoders();
	    Steering_ToUS(1);
	    HAL_Delay(200);  // center steering

	    Gyro_ResetHeading();
	    HAL_Delay(200);
	    Gyro_ResetHeading();

	    static float robot_x = 0.0f;
	    static float robot_y = 0.0f;
	    static float robot_heading_deg = 0.0f;

	    float delta_left_cm = 0.0f;
	    float delta_right_cm = 0.0f;
	    float d_center_cm = 0.0f;

	    float avg_cm =0.0f;


	    float prev_cm = 0.0f;
	    float prev_left_enc = 0.0f;
	    float prev_right_enc = 0.0f;
	    float v_ref_cm_s = 0.0f;

	    const float VMAX_CM_S = 100.0f;
	    const float ACC_CM_S2 = 30.0f;
	    const float CRUISE_CM_S = 30.0f;

	    const float FWD_KP = 0.18f; //0.18
	    const float FWD_KI = 0.02f; //0.03
	    const float FWD_KD = 0.0f;
	    const float FWD_I_CLAMP = 4000.0f;
	    const float FWD_CORR_CLAMP = 1200.0f;

	    const float WALL_KP = 8.0f; //500
	    const float WALL_KI = 0.00f; //0.04
	    const float WALL_KD = 1.00f; //0.08
	    const float WALL_I_CLAMP = 300.0f;
	    const float WALL_HEADING_BIAS_MAX_DEG = 1000.0f;
	    const float WALL_VALID_MIN_CM = 10.0f;
	    const float WALL_VALID_MAX_CM = 50.0f;

	    float l_v_integral = 0.0f, r_v_integral = 0.0f, a_integral = 0.0f;
	    float last_l_v_error = 0.0f, last_r_v_error = 0.0f, last_a_error = 0.0f;
	    float wall_i = 0.0f, wall_prev_err = 0.0f;
	    float wall_heading_bias_deg = 0.0f;

	    int hit_cnt = 0;
	    float right_ir_cm = 0.0f;
	    uint8_t right_ir_valid = 0u;

	    int prev_pwm_l = 0, prev_pwm_r = 0;

	    uint32_t last_time = HAL_GetTick();
	    uint32_t last_ir_ms = HAL_GetTick();
	    uint32_t last_oled_ms = HAL_GetTick();

	    char oled_line[32];

	    float WALL_TARGET_CM = 21.0f;
	    int ir_loop_count = 0;


	    // -----------------------------
	    // 2. Main Loop (Non-blocking)
	    // -----------------------------
	    while(1)
	    {
	        uint32_t now = HAL_GetTick();
	        float dt = (now - last_time) / 1000.0f;
	        if (dt <= 0.0f) dt = 0.001f;
	        last_time = now;

	        // -----------------------------
	        // a) Read right IR every 40ms
	        // -----------------------------
	        if (now - last_ir_ms >= 40u)
	        {
	            float ir_dt_s = (now - last_ir_ms) / 1000.0f;
	            if (ir_dt_s <= 0.0f) ir_dt_s = 0.001f;
	            last_ir_ms = now;


	            right_ir_cm = (float)IR_Left_Read_try();
	            if (ir_loop_count == 8){
	            	WALL_TARGET_CM = right_ir_cm;
	            }
	            right_ir_valid = (right_ir_cm >= WALL_VALID_MIN_CM && right_ir_cm <= WALL_VALID_MAX_CM) ? 1u : 0u;

	            // ---- Safety stop
	            if (right_ir_cm >= 75.0f) hit_cnt++;
	            else hit_cnt = 0;

	            if (hit_cnt >= 2)
	            {
	                Motor_stop();
	                OLED_ShowString(0,36,(uint8_t*)"Triggered R>=TH");
	                OLED_Refresh_Gram();
	                return avg_cm;
	            }

	            // ---- Wall PID
	            if ((dir > 0) && right_ir_valid && ir_loop_count > 10)
	            {
	                wall_prev_err = wall_prev_err;
	                float wall_err = right_ir_cm - WALL_TARGET_CM;
	                wall_i += wall_err * ir_dt_s;
	                wall_i = clampf(wall_i, -WALL_I_CLAMP, WALL_I_CLAMP);
	                float wall_d = (wall_err - wall_prev_err)/ir_dt_s;
	                wall_prev_err = wall_err;

	                wall_heading_bias_deg = WALL_KP*wall_err + WALL_KI*wall_i + WALL_KD*wall_d;
	            }
	            else
	            {
	                wall_i = 0.0f;
	                wall_prev_err = 0.0f;
	                wall_heading_bias_deg = 0.0f;
	            }

	            ir_loop_count++;
	        }

	        // -----------------------------
	        // b) Read gyro
	        // -----------------------------
	        Gyro_UpdateFromIMU(dt);
	        float heading = Gyro_GetHeadingDeg();
	        float heading_target_deg = ((dir > 0) && right_ir_valid) ? wall_heading_bias_deg : 0.0f;

	        // -----------------------------
	        // c) Read encoders
	        // -----------------------------

	        float left_enc = left_ticks_signed();
	        float right_enc = right_ticks_signed();
	        float left_cm  = left_enc / COUNTS_PER_CM_L;
	        float right_cm = right_enc / COUNTS_PER_CM_R;
	        avg_cm = (fabsf(left_cm) + fabsf(right_cm))*0.5f;

	        float delta_left_enc = left_enc - prev_left_enc;
	        float delta_right_enc = right_enc - prev_right_enc;
	        prev_left_enc = left_enc;
	        prev_right_enc = right_enc;

	        float meas_l_enc_s = delta_left_enc/dt;
	        float meas_r_enc_s = delta_right_enc/dt;

	        float v_meas_cm_s = (avg_cm - prev_cm)/dt;
	        prev_cm = avg_cm;

	        delta_left_cm = delta_left_enc / COUNTS_PER_CM_L;
	        delta_right_cm = delta_right_enc / COUNTS_PER_CM_R;

	        Odometry_Update(delta_left_cm, delta_right_cm, heading);
	        // -----------------------------
	        // d) Speed ramp
	        // -----------------------------
	        float target_speed_cm_s = clampf(CRUISE_CM_S, 0.0f, VMAX_CM_S);
	        if (v_ref_cm_s < target_speed_cm_s) v_ref_cm_s += ACC_CM_S2*dt;
	        else if (v_ref_cm_s > target_speed_cm_s) v_ref_cm_s -= ACC_CM_S2*dt;
	        v_ref_cm_s = clampf(v_ref_cm_s, 0.0f, target_speed_cm_s);

	        // -----------------------------
	        // e) Forward speed PID
	        // -----------------------------
	        float target_l_enc_s = dir*v_ref_cm_s*COUNTS_PER_CM_L;
	        float target_r_enc_s = dir*v_ref_cm_s*COUNTS_PER_CM_R;

	        float l_err = target_l_enc_s - meas_l_enc_s;
	        l_v_integral += l_err*dt;
	        l_v_integral = clampf(l_v_integral, -FWD_I_CLAMP, FWD_I_CLAMP);
	        float l_deriv = (l_err - last_l_v_error)/dt;
	        last_l_v_error = l_err;
	        float corr_l = FWD_KP*l_err + FWD_KI*l_v_integral + FWD_KD*l_deriv;
	        corr_l = clampf(corr_l, -FWD_CORR_CLAMP, FWD_CORR_CLAMP);

	        float r_err = target_r_enc_s - meas_r_enc_s;
	        r_v_integral += r_err*dt;
	        r_v_integral = clampf(r_v_integral, -FWD_I_CLAMP, FWD_I_CLAMP);
	        float r_deriv = (r_err - last_r_v_error)/dt;
	        last_r_v_error = r_err;
	        float corr_r = FWD_KP*r_err + FWD_KI*r_v_integral + FWD_KD*r_deriv;
	        corr_r = clampf(corr_r, -FWD_CORR_CLAMP, FWD_CORR_CLAMP);

	        int pwm_ff = pwm_from_speed_cmps(v_ref_cm_s);
	        int pwm_l = clampi((int)lroundf(pwm_ff + corr_l), 0, pwmMax);
	        int pwm_r = clampi((int)lroundf(pwm_ff + corr_r), 0, pwmMax);

	        // -----------------------------
	        // f) Heading PID
	        // -----------------------------
	        float a_err = heading_target_deg - heading;
	        a_integral += a_err*dt;
	        float a_deriv = (a_err - last_a_error)/dt;
	        last_a_error = a_err;

	        int heading_correction = (int)(0*a_err + 0*a_integral);

	        // -----------------------------
	        // g) Motor mixing + smooth PWM
	        // -----------------------------
	        int left_cmd = dir*pwm_l - heading_correction - wall_heading_bias_deg;
	        int right_cmd = dir*pwm_r + heading_correction + wall_heading_bias_deg;

	        // PWM low-pass filter
	        left_cmd = prev_pwm_l*0.7f + left_cmd*0.3f;
	        right_cmd = prev_pwm_r*0.7f + right_cmd*0.3f;
	        prev_pwm_l = left_cmd;
	        prev_pwm_r = right_cmd;

	        // Saturate
	        left_cmd  = clampi(left_cmd, -pwmMax, pwmMax);
	        right_cmd = clampi(right_cmd, -pwmMax, pwmMax);

	        set_left_motor(left_cmd);
	        set_right_motor(right_cmd);

	        // -----------------------------
	        // h) OLED update every 100ms
	        // -----------------------------
	        if (now - last_oled_ms >= 100u)
	        {
	            last_oled_ms = now;
	            OLED_Clear();
	            snprintf(oled_line, sizeof(oled_line), "T:%.1f", WALL_TARGET_CM);
	            OLED_ShowString(0,0,(uint8_t*)oled_line);
	            snprintf(oled_line, sizeof(oled_line), "L:%d R:%d", left_cmd, right_cmd);
	            OLED_ShowString(0,12,(uint8_t*)oled_line);
	//            rintf(oled_line, sizeof(oled_line), "Vm:%.2f Vt:%.2f", v_meas_cm_s, v_ref_cm_s);
	//            OLED_ShowString(0,12,(uint8_t*)oled_line);
	            snprintf(oled_line, sizeof(oled_line), "r_ir:%.2f",  right_ir_cm);
	            OLED_ShowString(0,24,(uint8_t*)oled_line);
	            snprintf(oled_line, sizeof(oled_line), "Average_cm:%.2f", avg_cm);
	            OLED_ShowString(0,36,(uint8_t*)oled_line);
	            OLED_Refresh_Gram();
	        }
	    }

	    return avg_cm;
}

float Drive_Straight_ToCM_RightIR_See(int dir)//working code for right IR exceed as of 18/03/2026
{
    // -----------------------------
    // 1. Initialization
    // -----------------------------
    reset_encoders();
    Steering_ToUS(1);
    HAL_Delay(200);  // center steering

    Gyro_ResetHeading();
    HAL_Delay(200);
    Gyro_ResetHeading();

    static float robot_x = 0.0f;
    static float robot_y = 0.0f;
    static float robot_heading_deg = 0.0f;

    float delta_left_cm = 0.0f;
    float delta_right_cm = 0.0f;
    float d_center_cm = 0.0f;

    float avg_cm =0.0f;


    float prev_cm = 0.0f;
    float prev_left_enc = 0.0f;
    float prev_right_enc = 0.0f;
    float v_ref_cm_s = 0.0f;

    const float VMAX_CM_S = 40.0f;
    const float ACC_CM_S2 = 10.0f;
    const float CRUISE_CM_S = 20.0f;

    const float FWD_KP = 0.18f; //0.18
    const float FWD_KI = 0.02f; //0.03
    const float FWD_KD = 0.0f;
    const float FWD_I_CLAMP = 4000.0f;
    const float FWD_CORR_CLAMP = 1200.0f;

    float l_v_integral = 0.0f, r_v_integral = 0.0f, a_integral = 0.0f;
    float last_l_v_error = 0.0f, last_r_v_error = 0.0f, last_a_error = 0.0f;

    int hit_cnt = 0;
    float right_ir_cm = 0.0f;
    uint8_t right_ir_valid = 0u;

    int prev_pwm_l = 0, prev_pwm_r = 0;

    uint32_t last_time = HAL_GetTick();
    uint32_t last_ir_ms = HAL_GetTick();
    uint32_t last_oled_ms = HAL_GetTick();

    char oled_line[32];

    float WALL_TARGET_CM = 21.0f;
    float WALL_VALID_MIN_CM = 3.0f;
    float WALL_VALID_MAX_CM = 50.0f;
    int ir_loop_count = 0;


    // -----------------------------
    // 2. Main Loop (Non-blocking)
    // -----------------------------
    while(1)
    {
        uint32_t now = HAL_GetTick();
        float dt = (now - last_time) / 1000.0f;
        if (dt <= 0.0f) dt = 0.001f;
        last_time = now;

        // -----------------------------
        // a) Read right IR every 40ms
        // -----------------------------
        if (now - last_ir_ms >= 40u)
        {
            float ir_dt_s = (now - last_ir_ms) / 1000.0f;
            if (ir_dt_s <= 0.0f) ir_dt_s = 0.001f;
            last_ir_ms = now;


            right_ir_cm = (float)IR_Right_Read_try();
            if (ir_loop_count == 8){
            	WALL_TARGET_CM = right_ir_cm;
            }
            right_ir_valid = (right_ir_cm >= WALL_VALID_MIN_CM && right_ir_cm <= WALL_VALID_MAX_CM) ? 1u : 0u;

            // ---- Safety stop
            if (right_ir_valid) hit_cnt++;
            else hit_cnt = 0;

            if (hit_cnt >= 2)
            {
                Motor_stop();
                OLED_ShowString(0,36,(uint8_t*)"Triggered R>=TH");
                OLED_Refresh_Gram();
                return avg_cm;
            }

            ir_loop_count++;
        }

        // -----------------------------
        // b) Read gyro
        // -----------------------------
        Gyro_UpdateFromIMU(dt);
        float heading = Gyro_GetHeadingDeg();

        // -----------------------------
        // c) Read encoders
        // -----------------------------

        float left_enc = left_ticks_signed();
        float right_enc = right_ticks_signed();
        float left_cm  = left_enc / COUNTS_PER_CM_L;
        float right_cm = right_enc / COUNTS_PER_CM_R;
        avg_cm = (fabsf(left_cm) + fabsf(right_cm))*0.5f;

        float delta_left_enc = left_enc - prev_left_enc;
        float delta_right_enc = right_enc - prev_right_enc;
        prev_left_enc = left_enc;
        prev_right_enc = right_enc;

        float meas_l_enc_s = delta_left_enc/dt;
        float meas_r_enc_s = delta_right_enc/dt;

        float v_meas_cm_s = (avg_cm - prev_cm)/dt;
        prev_cm = avg_cm;

        delta_left_cm = delta_left_enc / COUNTS_PER_CM_L;
        delta_right_cm = delta_right_enc / COUNTS_PER_CM_R;

        Odometry_Update(delta_left_cm, delta_right_cm, heading);
        // -----------------------------
        // d) Speed ramp
        // -----------------------------
        float target_speed_cm_s = clampf(CRUISE_CM_S, 0.0f, VMAX_CM_S);
        if (v_ref_cm_s < target_speed_cm_s) v_ref_cm_s += ACC_CM_S2*dt;
        else if (v_ref_cm_s > target_speed_cm_s) v_ref_cm_s -= ACC_CM_S2*dt;
        v_ref_cm_s = clampf(v_ref_cm_s, 0.0f, target_speed_cm_s);

        // -----------------------------
        // e) Forward speed PID
        // -----------------------------
        float target_l_enc_s = dir*v_ref_cm_s*COUNTS_PER_CM_L;
        float target_r_enc_s = dir*v_ref_cm_s*COUNTS_PER_CM_R;

        float l_err = target_l_enc_s - meas_l_enc_s;
        l_v_integral += l_err*dt;
        l_v_integral = clampf(l_v_integral, -FWD_I_CLAMP, FWD_I_CLAMP);
        float l_deriv = (l_err - last_l_v_error)/dt;
        last_l_v_error = l_err;
        float corr_l = FWD_KP*l_err + FWD_KI*l_v_integral + FWD_KD*l_deriv;
        corr_l = clampf(corr_l, -FWD_CORR_CLAMP, FWD_CORR_CLAMP);

        float r_err = target_r_enc_s - meas_r_enc_s;
        r_v_integral += r_err*dt;
        r_v_integral = clampf(r_v_integral, -FWD_I_CLAMP, FWD_I_CLAMP);
        float r_deriv = (r_err - last_r_v_error)/dt;
        last_r_v_error = r_err;
        float corr_r = FWD_KP*r_err + FWD_KI*r_v_integral + FWD_KD*r_deriv;
        corr_r = clampf(corr_r, -FWD_CORR_CLAMP, FWD_CORR_CLAMP);

        int pwm_ff = pwm_from_speed_cmps(v_ref_cm_s);
        int pwm_l = clampi((int)lroundf(pwm_ff + corr_l), 0, pwmMax);
        int pwm_r = clampi((int)lroundf(pwm_ff + corr_r), 0, pwmMax);

        // -----------------------------
        // f) Heading PID
        // -----------------------------
        float a_err = heading;
        a_integral += a_err*dt;
        float a_deriv = (a_err - last_a_error)/dt;
        last_a_error = a_err;

        int heading_correction = (int)(10*a_err) + (0*a_integral);

        // -----------------------------
        // g) Motor mixing + smooth PWM
        // -----------------------------
        int left_cmd = dir*pwm_l - heading_correction;
        int right_cmd = dir*pwm_r + heading_correction;

        // PWM low-pass filter
        left_cmd = prev_pwm_l*0.7f + left_cmd*0.3f;
        right_cmd = prev_pwm_r*0.7f + right_cmd*0.3f;
        prev_pwm_l = left_cmd;
        prev_pwm_r = right_cmd;

        // Saturate
        left_cmd  = clampi(left_cmd, -pwmMax, pwmMax);
        right_cmd = clampi(right_cmd, -pwmMax, pwmMax);

        set_left_motor(left_cmd);
        set_right_motor(right_cmd);

        // -----------------------------
        // h) OLED update every 100ms
        // -----------------------------
        if (now - last_oled_ms >= 100u)
        {
            last_oled_ms = now;
            OLED_Clear();
            snprintf(oled_line, sizeof(oled_line), "T:%.1f", WALL_TARGET_CM);
            OLED_ShowString(0,0,(uint8_t*)oled_line);
            snprintf(oled_line, sizeof(oled_line), "L:%d R:%d", left_cmd, right_cmd);
            OLED_ShowString(0,12,(uint8_t*)oled_line);
//            rintf(oled_line, sizeof(oled_line), "Vm:%.2f Vt:%.2f", v_meas_cm_s, v_ref_cm_s);
//            OLED_ShowString(0,12,(uint8_t*)oled_line);
            snprintf(oled_line, sizeof(oled_line), "r_ir:%.2f",  right_ir_cm);
            OLED_ShowString(0,24,(uint8_t*)oled_line);
            snprintf(oled_line, sizeof(oled_line), "Average_cm:%.2f", avg_cm);
            OLED_ShowString(0,36,(uint8_t*)oled_line);
            OLED_Refresh_Gram();
        }
    }

    return avg_cm;
}


float Drive_Straight_ToCM_LeftIR_See(int dir)//working code for right IR exceed as of 18/03/2026
{
    // -----------------------------
    // 1. Initialization
    // -----------------------------
    reset_encoders();
    Steering_ToUS(1);
    HAL_Delay(200);  // center steering

    Gyro_ResetHeading();
    HAL_Delay(200);
    Gyro_ResetHeading();

    static float robot_x = 0.0f;
    static float robot_y = 0.0f;
    static float robot_heading_deg = 0.0f;

    float delta_left_cm = 0.0f;
    float delta_right_cm = 0.0f;
    float d_center_cm = 0.0f;

    float avg_cm =0.0f;


    float prev_cm = 0.0f;
    float prev_left_enc = 0.0f;
    float prev_right_enc = 0.0f;
    float v_ref_cm_s = 0.0f;

    const float VMAX_CM_S = 40.0f;
    const float ACC_CM_S2 = 10.0f;
    const float CRUISE_CM_S = 20.0f;

    const float FWD_KP = 0.18f; //0.18
    const float FWD_KI = 0.02f; //0.03
    const float FWD_KD = 0.0f;
    const float FWD_I_CLAMP = 4000.0f;
    const float FWD_CORR_CLAMP = 1200.0f;

    float l_v_integral = 0.0f, r_v_integral = 0.0f, a_integral = 0.0f;
    float last_l_v_error = 0.0f, last_r_v_error = 0.0f, last_a_error = 0.0f;

    int hit_cnt = 0;
    float right_ir_cm = 0.0f;
    uint8_t right_ir_valid = 0u;

    int prev_pwm_l = 0, prev_pwm_r = 0;

    uint32_t last_time = HAL_GetTick();
    uint32_t last_ir_ms = HAL_GetTick();
    uint32_t last_oled_ms = HAL_GetTick();

    char oled_line[32];

    float WALL_TARGET_CM = 21.0f;
    float WALL_VALID_MIN_CM = 3.0f;
    float WALL_VALID_MAX_CM = 50.0f;
    int ir_loop_count = 0;


    // -----------------------------
    // 2. Main Loop (Non-blocking)
    // -----------------------------
    while(1)
    {
        uint32_t now = HAL_GetTick();
        float dt = (now - last_time) / 1000.0f;
        if (dt <= 0.0f) dt = 0.001f;
        last_time = now;

        // -----------------------------
        // a) Read right IR every 40ms
        // -----------------------------
        if (now - last_ir_ms >= 40u)
        {
            float ir_dt_s = (now - last_ir_ms) / 1000.0f;
            if (ir_dt_s <= 0.0f) ir_dt_s = 0.001f;
            last_ir_ms = now;


            right_ir_cm = (float)IR_Left_Read_try();
            if (ir_loop_count == 8){
            	WALL_TARGET_CM = right_ir_cm;
            }
            right_ir_valid = (right_ir_cm >= WALL_VALID_MIN_CM && right_ir_cm <= WALL_VALID_MAX_CM) ? 1u : 0u;

            // ---- Safety stop
            if (right_ir_valid) hit_cnt++;
            else hit_cnt = 0;

            if (hit_cnt >= 2)
            {
                Motor_stop();
                OLED_ShowString(0,36,(uint8_t*)"Triggered R>=TH");
                OLED_Refresh_Gram();
                return avg_cm;
            }

            ir_loop_count++;
        }

        // -----------------------------
        // b) Read gyro
        // -----------------------------
        Gyro_UpdateFromIMU(dt);
        float heading = Gyro_GetHeadingDeg();

        // -----------------------------
        // c) Read encoders
        // -----------------------------

        float left_enc = left_ticks_signed();
        float right_enc = right_ticks_signed();
        float left_cm  = left_enc / COUNTS_PER_CM_L;
        float right_cm = right_enc / COUNTS_PER_CM_R;
        avg_cm = (fabsf(left_cm) + fabsf(right_cm))*0.5f;

        float delta_left_enc = left_enc - prev_left_enc;
        float delta_right_enc = right_enc - prev_right_enc;
        prev_left_enc = left_enc;
        prev_right_enc = right_enc;

        float meas_l_enc_s = delta_left_enc/dt;
        float meas_r_enc_s = delta_right_enc/dt;

        float v_meas_cm_s = (avg_cm - prev_cm)/dt;
        prev_cm = avg_cm;

        delta_left_cm = delta_left_enc / COUNTS_PER_CM_L;
        delta_right_cm = delta_right_enc / COUNTS_PER_CM_R;

        Odometry_Update(delta_left_cm, delta_right_cm, heading);
        // -----------------------------
        // d) Speed ramp
        // -----------------------------
        float target_speed_cm_s = clampf(CRUISE_CM_S, 0.0f, VMAX_CM_S);
        if (v_ref_cm_s < target_speed_cm_s) v_ref_cm_s += ACC_CM_S2*dt;
        else if (v_ref_cm_s > target_speed_cm_s) v_ref_cm_s -= ACC_CM_S2*dt;
        v_ref_cm_s = clampf(v_ref_cm_s, 0.0f, target_speed_cm_s);

        // -----------------------------
        // e) Forward speed PID
        // -----------------------------
        float target_l_enc_s = dir*v_ref_cm_s*COUNTS_PER_CM_L;
        float target_r_enc_s = dir*v_ref_cm_s*COUNTS_PER_CM_R;

        float l_err = target_l_enc_s - meas_l_enc_s;
        l_v_integral += l_err*dt;
        l_v_integral = clampf(l_v_integral, -FWD_I_CLAMP, FWD_I_CLAMP);
        float l_deriv = (l_err - last_l_v_error)/dt;
        last_l_v_error = l_err;
        float corr_l = FWD_KP*l_err + FWD_KI*l_v_integral + FWD_KD*l_deriv;
        corr_l = clampf(corr_l, -FWD_CORR_CLAMP, FWD_CORR_CLAMP);

        float r_err = target_r_enc_s - meas_r_enc_s;
        r_v_integral += r_err*dt;
        r_v_integral = clampf(r_v_integral, -FWD_I_CLAMP, FWD_I_CLAMP);
        float r_deriv = (r_err - last_r_v_error)/dt;
        last_r_v_error = r_err;
        float corr_r = FWD_KP*r_err + FWD_KI*r_v_integral + FWD_KD*r_deriv;
        corr_r = clampf(corr_r, -FWD_CORR_CLAMP, FWD_CORR_CLAMP);

        int pwm_ff = pwm_from_speed_cmps(v_ref_cm_s);
        int pwm_l = clampi((int)lroundf(pwm_ff + corr_l), 0, pwmMax);
        int pwm_r = clampi((int)lroundf(pwm_ff + corr_r), 0, pwmMax);

        // -----------------------------
        // f) Heading PID
        // -----------------------------
        float a_err = heading;
        a_integral += a_err*dt;
        float a_deriv = (a_err - last_a_error)/dt;
        last_a_error = a_err;

        int heading_correction = (int)(10*a_err) + (0*a_integral);

        // -----------------------------
        // g) Motor mixing + smooth PWM
        // -----------------------------
        int left_cmd = dir*pwm_l - heading_correction;
        int right_cmd = dir*pwm_r + heading_correction;

        // PWM low-pass filter
        left_cmd = prev_pwm_l*0.7f + left_cmd*0.3f;
        right_cmd = prev_pwm_r*0.7f + right_cmd*0.3f;
        prev_pwm_l = left_cmd;
        prev_pwm_r = right_cmd;

        // Saturate
        left_cmd  = clampi(left_cmd, -pwmMax, pwmMax);
        right_cmd = clampi(right_cmd, -pwmMax, pwmMax);

        set_left_motor(left_cmd);
        set_right_motor(right_cmd);

        // -----------------------------
        // h) OLED update every 100ms
        // -----------------------------
        if (now - last_oled_ms >= 100u)
        {
            last_oled_ms = now;
            OLED_Clear();
            snprintf(oled_line, sizeof(oled_line), "T:%.1f", WALL_TARGET_CM);
            OLED_ShowString(0,0,(uint8_t*)oled_line);
            snprintf(oled_line, sizeof(oled_line), "L:%d R:%d", left_cmd, right_cmd);
            OLED_ShowString(0,12,(uint8_t*)oled_line);
//            rintf(oled_line, sizeof(oled_line), "Vm:%.2f Vt:%.2f", v_meas_cm_s, v_ref_cm_s);
//            OLED_ShowString(0,12,(uint8_t*)oled_line);
            snprintf(oled_line, sizeof(oled_line), "r_ir:%.2f",  right_ir_cm);
            OLED_ShowString(0,24,(uint8_t*)oled_line);
            snprintf(oled_line, sizeof(oled_line), "Average_cm:%.2f", avg_cm);
            OLED_ShowString(0,36,(uint8_t*)oled_line);
            OLED_Refresh_Gram();
        }
    }

    return avg_cm;
}

void Drive_Straight_ToCM_Left_IR (float target_cm)
{
    // -----------------------------
    // 1. Initialization
    // -----------------------------
    reset_encoders();

    Steering_ToUS(30);
    HAL_Delay(500);
    Steering_ToUS(0);
    HAL_Delay(500);

    Gyro_ResetHeading();
    HAL_Delay(200);
    Gyro_ResetHeading();

    float prev_cm = 0.0f;
    float prev_left_enc = 0.0f;
    float prev_right_enc = 0.0f;
    float v_ref_cm_s = 0.0f;

    const float STOP_TOL_CM = 2.0f;
    const uint32_t OBSTACLE_STOP_CM = 20;

    // --- NEW: left IR stop settings ---
    const int LEFT_IR_STOP_CM = 40;      // tune this
    const int LEFT_IR_CONSEC_NEEDED = 3; // debounce count
    uint8_t left_ir_hit_cnt = 0;

    const float VMAX_CM_S = 80.0f;
    const float ACC_CM_S2 = 40.0f;

    const float FWD_KP = 0.18f;
    const float FWD_KI = 0.03f;
    const float FWD_KD = 0.0f;
    const float FWD_I_CLAMP = 4000.0f;
    const float FWD_CORR_CLAMP = 1200.0f;

    const int dir = (target_cm >= 0.0f) ? +1 : -1;
    target_cm = fabsf(target_cm);

    float a_integral = 0.0f;
    float l_v_integral = 0.0f;
    float r_v_integral = 0.0f;
    float last_a_error = 0.0f;
    float last_l_v_error = 0.0f;
    float last_r_v_error = 0.0f;

    uint32_t last_time = HAL_GetTick();
    uint32_t last_oled = HAL_GetTick();
    char oled_line[32];

    // -----------------------------
    // 2. Main Loop
    // -----------------------------
    while (1)
    {
        // ----- Emergency stop for ultrasonic obstacle -----
        uint32_t distance_cm = HCSR04_Read();
        if (distance_cm > 0 && distance_cm <= OBSTACLE_STOP_CM)
        {
            Motor_stop();
            OLED_Clear();
            OLED_ShowString(0, 0, (uint8_t*)"EMERGENCY STOP");
            snprintf(oled_line, sizeof(oled_line), "US Dist:%lu cm", (unsigned long)distance_cm);
            OLED_ShowString(0, 16, (uint8_t*)oled_line);
            OLED_ShowString(0, 32, (uint8_t*)"Obstacle detected!");
            OLED_Refresh_Gram();
            break;
        }

        // ----- NEW: stop if left IR detects object -----
        IR_Left_Read();

        if (distanceir <= LEFT_IR_STOP_CM)
        {
            if (left_ir_hit_cnt < LEFT_IR_CONSEC_NEEDED) left_ir_hit_cnt++;
        }
        else
        {
            left_ir_hit_cnt = 0;
        }

        if (left_ir_hit_cnt >= LEFT_IR_CONSEC_NEEDED)
        {
            Motor_stop();
            OLED_Clear();
            OLED_ShowString(0, 0, (uint8_t*)"LEFT IR STOP");
            snprintf(oled_line, sizeof(oled_line), "L IR:%d cm", distanceir);
            OLED_ShowString(0, 16, (uint8_t*)oled_line);
            OLED_ShowString(0, 32, (uint8_t*)"Left object seen");
            OLED_Refresh_Gram();
            break;
        }

        // ----- Time Step -----
        uint32_t now = HAL_GetTick();
        float dt = (now - last_time) / 1000.0f;
        if (dt <= 0.0f) dt = 0.001f;
        last_time = now;

        // ----- Update gyro -----
        Gyro_UpdateFromIMU(dt);
        float heading = Gyro_GetHeadingDeg();   // +left, -right

        // ----- Distance from encoders -----
        float left_enc  = left_ticks_signed();
        float right_enc = right_ticks_signed();

        float left_cm  = left_enc  / COUNTS_PER_CM_L;
        float right_cm = right_enc / COUNTS_PER_CM_R;

        float delta_left_enc  = left_enc  - prev_left_enc;
        float delta_right_enc = right_enc - prev_right_enc;

        float meas_l_enc_s = delta_left_enc / dt;
        float meas_r_enc_s = delta_right_enc / dt;

        prev_left_enc = left_enc;
        prev_right_enc = right_enc;

        float avg_cm = (fabsf(left_cm) + fabsf(right_cm)) * 0.5f;
        float v_meas_cm_s = (avg_cm - prev_cm) / dt;


// -----------------------------
        // 3. Speed Profile (Soft Braking)
        // -----------------------------
        float S = target_cm - avg_cm;
        float Ssv = S - (v_ref_cm_s * dt);
        float Sbr = (v_ref_cm_s * v_ref_cm_s) / (2.0f * ACC_CM_S2);
        float v_next = v_ref_cm_s + ACC_CM_S2 * dt;
        float Siv = S - (v_next * dt);
        float Sbriv = (v_next * v_next) / (2.0f * ACC_CM_S2);

        if (S < 0.0f || Ssv < Sbr)
        {
            v_ref_cm_s -= ACC_CM_S2 * dt;
        }
        else if (Siv > Sbriv && v_ref_cm_s < VMAX_CM_S)
        {
            v_ref_cm_s += ACC_CM_S2 * dt;
        }

        v_ref_cm_s = clampf(v_ref_cm_s, 0.0f, VMAX_CM_S);
        prev_cm = avg_cm;

        // -----------------------------
        // 4. Forward speed PID (enc/s)
        // -----------------------------
        float target_l_enc_s = (float)dir * v_ref_cm_s * COUNTS_PER_CM_L;
        float target_r_enc_s = (float)dir * v_ref_cm_s * COUNTS_PER_CM_R;

        float l_v_err = target_l_enc_s - meas_l_enc_s;
        l_v_integral += l_v_err * dt;
        l_v_integral = clampf(l_v_integral, -FWD_I_CLAMP, FWD_I_CLAMP);
        float l_v_derivative = (l_v_err - last_l_v_error) / dt;
        last_l_v_error = l_v_err;
        float corr_l = (FWD_KP * l_v_err) + (FWD_KI * l_v_integral) + (FWD_KD * l_v_derivative);
        corr_l = clampf(corr_l, -FWD_CORR_CLAMP, FWD_CORR_CLAMP);

        float r_v_err = target_r_enc_s - meas_r_enc_s;
        r_v_integral += r_v_err * dt;
        r_v_integral = clampf(r_v_integral, -FWD_I_CLAMP, FWD_I_CLAMP);
        float r_v_derivative = (r_v_err - last_r_v_error) / dt;
        last_r_v_error = r_v_err;
        float corr_r = (FWD_KP * r_v_err) + (FWD_KI * r_v_integral) + (FWD_KD * r_v_derivative);
        corr_r = clampf(corr_r, -FWD_CORR_CLAMP, FWD_CORR_CLAMP);

        int pwm_ff = pwm_from_speed_cmps(v_ref_cm_s);
        int pwm_l = clampi((int)lroundf((float)pwm_ff + corr_l), 0, pwmMax);
        int pwm_r = clampi((int)lroundf((float)pwm_ff + corr_r), 0, pwmMax);

        // -----------------------------
        // 5. Gyro PID
        // -----------------------------
        float a_error = -heading;
        a_integral += a_error * dt;
        float a_differential = (a_error - last_a_error) / dt;
        last_a_error = a_error;

        int heading_correction =
            (int)(1500.0f * a_error +
                  100.0f * a_integral +
                  0.0f * a_differential);

        // -----------------------------
        // 6. OLED debug
        // -----------------------------
        if (now - last_oled >= 100u)
        {
            last_oled = now;

            OLED_Clear();

            snprintf(oled_line, sizeof(oled_line), "D:%.2f R:%.2f", avg_cm, target_cm - avg_cm);
            OLED_ShowString(0, 0, (uint8_t*)oled_line);

            snprintf(oled_line, sizeof(oled_line), "Vm:%.2f Vt:%.2f", v_meas_cm_s, v_ref_cm_s);
            OLED_ShowString(0, 12, (uint8_t*)oled_line);

            snprintf(oled_line, sizeof(oled_line), "Head:%.2f", heading);
            OLED_ShowString(0, 24, (uint8_t*)oled_line);

            snprintf(oled_line, sizeof(oled_line), "LIR:%d cnt:%d", distanceir, left_ir_hit_cnt);
            OLED_ShowString(0, 36, (uint8_t*)oled_line);

            snprintf(oled_line, sizeof(oled_line), "Hcor:%d", heading_correction);
            OLED_ShowString(0, 48, (uint8_t*)oled_line);

            OLED_Refresh_Gram();
        }

        // -----------------------------
        // 7. Motor Mixing
        // -----------------------------
        int left_cmd  = dir * pwm_l - heading_correction;
        int right_cmd = dir * pwm_r + heading_correction;

        if (left_cmd  > pwmMax)  left_cmd  = pwmMax;
        if (left_cmd  < -pwmMax) left_cmd  = -pwmMax;
        if (right_cmd > pwmMax)  right_cmd = pwmMax;
        if (right_cmd < -pwmMax) right_cmd = -pwmMax;

        set_left_motor(left_cmd);
        set_right_motor(right_cmd);

        HAL_Delay(10);


// ----- Normal distance stop -----
        if (target_cm - avg_cm <= STOP_TOL_CM)
        {
            break;
        }
    }

    // -----------------------------
    // 8. Stop Motors
    // -----------------------------
    Motor_stop();
}


void Drive_Forward_Until_LeftIR_detects(int forward_pwm,
                                        int threshold_cm,
                                        int consec_needed,
                                        uint32_t poll_ms)
{
    if (forward_pwm < pwmMin) forward_pwm = pwmMin;
    if (forward_pwm > pwmMax) forward_pwm = pwmMax;
    if (threshold_cm < 1)     threshold_cm = 1;
    if (consec_needed < 1)    consec_needed = 1;
    if (poll_ms < 20)         poll_ms = 20;

    Steering_ToUS(0);           // wheels straight
    HAL_Delay(30);
    Motor_forward_PID(forward_pwm);

    uint8_t hit_cnt = 0;
    uint32_t last = HAL_GetTick();

    OLED_ShowString(0, 36, (uint8_t*)"Fwd until L>=TH      ");
    OLED_Refresh_Gram();

    for (;;) {
        uint32_t now = HAL_GetTick();
        if (now - last >= poll_ms) {
            last = now;

            // Read left IR (and right if you want to show it)
            IR_Left_Read();      // updates: distanceir (cm), iDistanceL
            IR_Right_Read();     // optional, for display

            // OLED debug (optional)
            char lbuf[32];
            snprintf(lbuf, sizeof(lbuf), "IR L:%3dcm R:%3dcm   ", distanceir, distanceir_right);
            OLED_ShowString(0, 0,  (uint8_t*)lbuf);
            char rbuf[32];
            snprintf(rbuf, sizeof(rbuf), "RawL:%4d RawR:%4d   ", iDistanceL, iDistanceR);
            OLED_ShowString(0, 12, (uint8_t*)rbuf);
            OLED_Refresh_Gram();

            // Debounced threshold
            if (distanceir <= threshold_cm) {
                if (hit_cnt < consec_needed) hit_cnt++;
            } else {
                hit_cnt = 0;
            }

            if (hit_cnt >= consec_needed) {
                Motor_stop();
                OLED_ShowString(0, 36, (uint8_t*)"Triggered: L>=TH     ");
                OLED_Refresh_Gram();
                HAL_Delay(100);
                return; // done
            }
        }

        // keep loop snappy
        HAL_Delay(5);
    }
}

// Convenience wrapper: after the stop, make a LEFT turn and resume forward
void Drive_Forward_Until_LeftIR_Then_LeftTurn(int forward_pwm,
                                              int threshold_cm,
                                              int consec_needed,
                                              uint32_t poll_ms,
                                              int turn_deg,   // e.g., 90
                                              int steer_deg)  // e.g., -30 for left
{
    Drive_Forward_Until_LeftIR_Exceeds(forward_pwm, threshold_cm, consec_needed, poll_ms);

    // Do the left turn
    OLED_ShowString(0, 36, (uint8_t*)"Turning LEFT...      ");
    OLED_Refresh_Gram();
    HAL_Delay(80);

    Drive_Turn_Angle((float)turn_deg, 0);

}

void Drive_Forward_Until_RightIR_Exceeds(int forward_pwm,
                                         int threshold_cm,
                                         int consec_needed,
                                         uint32_t poll_ms)
{
    if (forward_pwm < pwmMin) forward_pwm = pwmMin;
    if (forward_pwm > pwmMax) forward_pwm = pwmMax;
    if (threshold_cm < 1)     threshold_cm = 1;
    if (consec_needed < 1)    consec_needed = 1;
    if (poll_ms < 20)         poll_ms = 20;

    Steering_ToUS(0);     // wheels straight
    HAL_Delay(30);
    Motor_forward_PID(forward_pwm);

    uint8_t  hit_cnt = 0;
    uint32_t last    = HAL_GetTick();

    OLED_ShowString(0, 36, (uint8_t*)"Fwd until R>=TH      ");
    OLED_Refresh_Gram();

    for (;;) {
        uint32_t now = HAL_GetTick();
        if (now - last >= poll_ms) {
            last = now;

            IR_Left_Read();
            IR_Right_Read();

            char lbuf[32];
            snprintf(lbuf, sizeof(lbuf), "IR L:%3dcm R:%3dcm   ", distanceir, distanceir_right);
            OLED_ShowString(0, 0,  (uint8_t*)lbuf);
            char rbuf[32];
            snprintf(rbuf, sizeof(rbuf), "RawL:%4d RawR:%4d   ", iDistanceL, iDistanceR);
            OLED_ShowString(0, 12, (uint8_t*)rbuf);
            OLED_Refresh_Gram();

            // Debounced threshold
            if (distanceir_right >= threshold_cm) {
                if (hit_cnt < consec_needed) hit_cnt++;
            } else {
                hit_cnt = 0;
            }

            if (hit_cnt >= consec_needed) {
                Motor_stop();
                OLED_ShowString(0, 36, (uint8_t*)"Triggered: R>=TH     ");
                OLED_Refresh_Gram();
                HAL_Delay(100);
                return; // done
            }
        }
        HAL_Delay(5);
    }
}

void Drive_Forward_Until_RightIR_detects(int forward_pwm,
                                         int threshold_cm,
                                         int consec_needed,
                                         uint32_t poll_ms)
{
    if (forward_pwm < pwmMin) forward_pwm = pwmMin;
    if (forward_pwm > pwmMax) forward_pwm = pwmMax;
    if (threshold_cm < 1)     threshold_cm = 1;
    if (consec_needed < 1)    consec_needed = 1;
    if (poll_ms < 20)         poll_ms = 20;

    Steering_ToUS(0);
    HAL_Delay(30);
    Motor_forward_PID(forward_pwm);

    uint8_t  hit_cnt = 0;
    uint32_t last    = HAL_GetTick();

    OLED_ShowString(0, 36, (uint8_t*)"Fwd until R<=TH      ");
    OLED_Refresh_Gram();

    for (;;) {
        uint32_t now = HAL_GetTick();
        if (now - last >= poll_ms) {
            last = now;

            IR_Left_Read();
            IR_Right_Read();

            char lbuf[32];
            snprintf(lbuf, sizeof(lbuf), "IR L:%3dcm R:%3dcm   ", distanceir, distanceir_right);
            OLED_ShowString(0, 0,  (uint8_t*)lbuf);
            char rbuf[32];
            snprintf(rbuf, sizeof(rbuf), "RawL:%4d RawR:%4d   ", iDistanceL, iDistanceR);
            OLED_ShowString(0, 12, (uint8_t*)rbuf);
            OLED_Refresh_Gram();

            // RIGHT <= threshold
            if (distanceir_right <= threshold_cm) {
                if (hit_cnt < consec_needed) hit_cnt++;
            } else {
                hit_cnt = 0;
            }

            if (hit_cnt >= consec_needed) {
                Motor_stop();
                OLED_ShowString(0, 36, (uint8_t*)"Triggered: R<=TH     ");
                OLED_Refresh_Gram();
                HAL_Delay(100);
                return;
            }
        }
        HAL_Delay(5);
    }
}

void Drive_Forward_Until_RightIR_Then_RightTurn(int forward_pwm,
                                                int threshold_cm,
                                                int consec_needed,
                                                uint32_t poll_ms,
                                                int turn_deg,   // e.g., 90
                                                int steer_deg)  // e.g., +30 for right
{
    // mirror of left version — trigger when RIGHT IR exceeds threshold
    Drive_Forward_Until_RightIR_Exceeds(forward_pwm, threshold_cm, consec_needed, poll_ms);

    OLED_ShowString(0, 36, (uint8_t*)"Turning RIGHT...     ");
    OLED_Refresh_Gram();
    HAL_Delay(80);

    Drive_Turn_Angle((float)turn_deg, 0);

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

// Signed turn helper: backward=0 forward entry, backward=1 reverse entry.
// Forward Right: Turn_Dir(-ve deg, 0)
// Forward Left: Turn_Dir(+ve deg, 0)
// Backward Left: Turn_Dir(+ve deg, 1)
// Backward Right: Turn_Dir(-ve deg, 1)

static float Drive_Turn_Angle(float target_turn_deg, int backward) // james code, allows reverse movement error correction
{
    if (fabsf(target_turn_deg) < 0.5f)
    {
        Motor_stop();
        Steering_ToUS(0);
        return 0.0f;
    }

    // Fixed steer command of +/-30 maps to about 25 real steering degrees.
    const int16_t STEER_CMD = 30;
    const float STOP_TOL_DEG = 1.0f;
    const float STOP_YAW_DPS = 4.0f;
    const uint32_t STOP_HOLD_MS = 120u;
    const float KP_TURN = 0.8f;
    const float KD_TURN = 0.8f;
    const float OMEGA_MAX_DPS = 85.0f;
    const float ALPHA_DPS2 = 180.0f;
    const int PWM_TURN_MIN = 1400;
    const int PWM_TURN_MAX = 2200;
    const float WHEELBASE_CM = 14.5f;
    const float TRACK_CM = 16.1f;
    const float STEER_REAL_DEG = 25.0f;
    const float RAD_PER_DEG = 3.14159265f / 180.0f;
    const uint32_t TURN_TIMEOUT_MS = 8000u;
    const uint8_t USE_TURN_COMP = 0u;

    float cmd_target_deg = USE_TURN_COMP
        ? Apply_Turn_Compensation(target_turn_deg, backward)
        : target_turn_deg;
    const int turn_sign = (cmd_target_deg >= 0.0f) ? +1 : -1;
    const int steer_sign = backward ? -turn_sign : turn_sign;
    const int steer_right = (steer_sign > 0) ? 1 : 0;
    const int16_t target_steer = (steer_sign > 0) ? +STEER_CMD : (int16_t)(-STEER_CMD + 2);

    float phi = STEER_REAL_DEG * RAD_PER_DEG;
    float tan_phi = tanf(phi);
    float R = (fabsf(tan_phi) > 1e-5f) ? (WHEELBASE_CM / tan_phi) : 1e6f;
    float k = TRACK_CM / (2.0f * R);
    float outer_mul = 1.0f + k;
    float inner_mul = 1.0f - k;
    if (inner_mul < 0.0f) inner_mul = 0.0f;

    Steering_ToUS(30);
    HAL_Delay(500);
    Steering_ToUS(0);
    HAL_Delay(500);

    // steering to US 30 deg and -28deg ~~= 25 deg and -25 deg irl
    Steering_ToUS(target_steer);
    HAL_Delay(120);
    Gyro_ResetHeading();

    uint32_t last = HAL_GetTick();
    uint32_t start_ms = last;
    uint32_t last_oled = last;
    uint32_t tol_enter_ms = 0u;
    char oled_line[32];
    float heading_deg = 0.0f;
    float heading_rate_dps = 0.0f;
    float err_deg = cmd_target_deg;
    float omega_cmd = 0.0f;
    int move_dir = backward ? -1 : +1;
    int last_move_dir = move_dir;
    int exited_by_tol = 0;
    int timed_out = 0;

    while (1)
    {
        uint32_t now = HAL_GetTick();
        float dt = (now - last) / 1000.0f;
        if (dt <= 0.0f) dt = 0.001f;
        last = now;

        if ((now - start_ms) >= TURN_TIMEOUT_MS)
        {
            timed_out = 1;
            break;
        }

        Gyro_UpdateFromIMU(dt);

        heading_deg = -Gyro_GetHeadingDeg();
        heading_rate_dps = -gz_dps;
        err_deg = cmd_target_deg - heading_deg;
        float rem_deg = fabsf(err_deg);
        float omega_cap = sqrtf(2.0f * ALPHA_DPS2 * rem_deg);
        if (omega_cap > OMEGA_MAX_DPS) omega_cap = OMEGA_MAX_DPS;
        if (rem_deg < 3.0f && omega_cap > 20.0f) omega_cap = 20.0f;

        omega_cmd = (KP_TURN * err_deg) - (KD_TURN * heading_rate_dps);
        omega_cmd = clampf(omega_cmd, -omega_cap, omega_cap);

        int yaw_sign = 0;
        if (omega_cmd > 0.0f) yaw_sign = +1;
        else if (omega_cmd < 0.0f) yaw_sign = -1;
        else yaw_sign = (err_deg >= 0.0f) ? +1 : -1;
        int proposed_move_dir = yaw_sign * steer_sign;
        if ((proposed_move_dir != last_move_dir) && (rem_deg <= 2.0f))
            move_dir = last_move_dir;
        else
        {
            move_dir = proposed_move_dir;
            last_move_dir = move_dir;
        }

        if ((rem_deg <= STOP_TOL_DEG) && (fabsf(heading_rate_dps) <= STOP_YAW_DPS))
        {
            if (tol_enter_ms == 0u) tol_enter_ms = now;
            if ((now - tol_enter_ms) >= STOP_HOLD_MS)
            {
                exited_by_tol = 1;
                break;
            }
        }
        else
        {
            tol_enter_ms = 0u;
        }

        float omega_ratio = (OMEGA_MAX_DPS > 0.0f) ? (fabsf(omega_cmd) / OMEGA_MAX_DPS) : 0.0f;
        float pwm_base_f = (float)PWM_TURN_MIN +
                           omega_ratio * (float)(PWM_TURN_MAX - PWM_TURN_MIN);
        int pwm_base = clampi((int)lroundf(pwm_base_f), 0, pwmMax);
        int pwm_outer = clampi((int)lroundf((float)pwm_base * outer_mul), 0, pwmMax);
        int pwm_inner = clampi((int)lroundf((float)pwm_base * inner_mul), 0, pwmMax);

        int left_mag = steer_right ? pwm_outer : pwm_inner;
        int right_mag = steer_right ? pwm_inner : pwm_outer;
        int left_cmd = move_dir * left_mag;
        int right_cmd = move_dir * right_mag;

        set_left_motor(left_cmd);
        set_right_motor(right_cmd);

        if (now - last_oled >= 100u)
        {
            last_oled = now;
            OLED_Clear();

            snprintf(oled_line, sizeof(oled_line), "Target:%.1f", cmd_target_deg);
            OLED_ShowString(0, 0, (uint8_t*)oled_line);

            snprintf(oled_line, sizeof(oled_line), "Heading:%.1f", heading_deg);
            OLED_ShowString(0, 12, (uint8_t*)oled_line);

            snprintf(oled_line, sizeof(oled_line), "Err:%.1f", err_deg);
            OLED_ShowString(0, 24, (uint8_t*)oled_line);

            snprintf(oled_line, sizeof(oled_line), "Rate:%.1f", heading_rate_dps);
            OLED_ShowString(0, 36, (uint8_t*)oled_line);

            snprintf(oled_line, sizeof(oled_line), "W:%.1f D:%c", omega_cmd, (move_dir >= 0) ? 'F' : 'R');
            OLED_ShowString(0, 48, (uint8_t*)oled_line);

            OLED_Refresh_Gram();
        }

        HAL_Delay(10);
    }

    Motor_stop();
    Steering_ToUS(0);
    HAL_Delay(100);
    float actual_signed = -Gyro_GetHeadingDeg();
    if (exited_by_tol)
        Update_Turn_Residual(target_turn_deg, actual_signed);

    OLED_Clear();
    snprintf(oled_line, sizeof(oled_line), "Target:%.1f", cmd_target_deg);
    OLED_ShowString(0, 0, (uint8_t*)oled_line);
    snprintf(oled_line, sizeof(oled_line), "Heading:%.1f", actual_signed);
    OLED_ShowString(0, 16, (uint8_t*)oled_line);
    snprintf(oled_line, sizeof(oled_line), "Err:%.1f", cmd_target_deg - actual_signed);
    OLED_ShowString(0, 32, (uint8_t*)oled_line);
    OLED_ShowString(0, 48, (uint8_t*)(timed_out ? "TIMEOUT" : "OK"));
    OLED_Refresh_Gram();

    return actual_signed;
}

float Drive_Arc_Turn(float target_deg, int steer_deg, int base_pwm)
{
    float prev_left = 0.0f;
    float prev_right = 0.0f;

    if (target_deg < 0.0f)
        target_deg = -target_deg;

    float target_heading = (steer_deg >= 0) ? -target_deg : target_deg;

    reset_encoders();
    Motor_stop();
    Steering_ToUS(STEER_CENTER + steer_deg);
    HAL_Delay(300);

    Gyro_ResetHeading();
    HAL_Delay(100);
    Gyro_ResetHeading();

    float integral = 0.0f;
    float last_error = 0.0f;

    const float ARC_KP = 12.0f;
    const float ARC_KI = 0.1f;
    const float ARC_KD = 1.7f;
    const uint32_t STOP_HOLD_MS = 100u;

    uint32_t last_time = HAL_GetTick();
    uint32_t tol_enter_ms = 0u;

    while (1)
    {
        uint32_t now = HAL_GetTick();
        float dt = (now - last_time) / 1000.0f;
        if (dt <= 0.0f) dt = 0.001f;
        last_time = now;

        Gyro_UpdateFromIMU(dt);
        float heading = Gyro_GetHeadingDeg();

        float error = target_heading - heading;
        float remaining = fabsf(error);
        if (remaining <= 1.0f) {
				Motor_stop(); //3 ~~2.8deg 5 ~~4.8deg 2 ~~1.8deg
				break;
		}

        // -----------------------------
        // Encoder tracking
        // -----------------------------
        float left_cm  = left_ticks_signed() / COUNTS_PER_CM_L;
        float right_cm = right_ticks_signed() / COUNTS_PER_CM_R;

        float d_left  = left_cm  - prev_left;
        float d_right = right_cm - prev_right;

        prev_left  = left_cm;
        prev_right = right_cm;

        Odometry_Update(d_left, d_right, Gyro_GetHeadingDeg());

        // -----------------------------
        // Motor control (unchanged)
        // -----------------------------
        int pwm = base_pwm;
        if      (remaining > 20.0f) pwm = base_pwm;
        else if (remaining > 10.0f) pwm = (int)(base_pwm * 0.55f);
        else if (remaining > 5.0f)  pwm = (int)(base_pwm * 0.45f);
        else                        pwm = (int)(base_pwm * 0.40f);
        if (pwm < 280) pwm = 280;

        integral += error * dt;
        integral = clampf(integral, -30.0f, 30.0f);

        float derivative = (error - last_error) / dt;
        last_error = error;

        int corr = (int)(ARC_KP * error + ARC_KI * integral + ARC_KD * derivative);
        int corr_limit = (int)(0.20f * pwm);
        if (corr_limit < 50) corr_limit = 50;
        corr = clampi(corr, -corr_limit, corr_limit);

        int left_base, right_base;

        if (steer_deg > 0 && target_deg >= 180.0f)
        {
            left_base  = (int)(pwm * 1.4f);
            right_base = pwm * 0.0;
        }
        else if (steer_deg > 0)
        {
            left_base  = (int)(pwm * 1.2f);
            right_base = pwm;
        }
        else
        {
            left_base  = pwm;
            right_base = (int)(pwm * 1.8f);
        }

        int left_cmd  = left_base  + corr;
        int right_cmd = right_base - corr;

        left_cmd  = clampi(left_cmd,  -pwmMax, pwmMax);
        right_cmd = clampi(right_cmd, -pwmMax, pwmMax);

        set_left_motor(left_cmd);
        set_right_motor(right_cmd);

        HAL_Delay(10);
    }

    Motor_stop();
    //HAL_Delay(80);
    //Steering_ToUS(STEER_CENTER);

    // -----------------------------
    // Final heading update
    // -----------------------------
    float final_heading = -Gyro_GetHeadingDeg();
    float actual_heading = -final_heading;
    float turn_error_deg = wrap180f(target_heading - actual_heading);
    robot_heading_deg += final_heading;
    if (robot_heading_deg > 180.0f)  robot_heading_deg -= 360.0f;
    if (robot_heading_deg < -180.0f) robot_heading_deg += 360.0f;

    // -----------------------------
    // 🔥 Final OLED display of position & heading
    // -----------------------------
    char pos_buf[32];
    OLED_Clear();
    snprintf(pos_buf, sizeof(pos_buf), "X:%.1f Y:%.1f", robot_x, robot_y);
    OLED_ShowString(0, 0, (uint8_t*)pos_buf);
    snprintf(pos_buf, sizeof(pos_buf), "H:%.1f", turn_error_deg);
    OLED_ShowString(0, 16, (uint8_t*)pos_buf);
    OLED_Refresh_Gram();
    HAL_Delay(30); // give time to read final position
    return turn_error_deg;
}


void Drive_Arc_Turn_Backward(float target_deg, int steer_deg, int base_pwm)
{
    char buf[32];

    if (target_deg < 0.0f)
        target_deg = -target_deg;

    // Same heading convention as your forward version
    float target_heading = (steer_deg >= 0) ? target_deg : -target_deg;

    reset_encoders();
    Motor_stop();
    Steering_ToUS(STEER_CENTER);

    // Apply steering (relative to center)
    Steering_ToUS(STEER_CENTER + steer_deg);
    HAL_Delay(300);

    // Reset gyro
    Gyro_ResetHeading();
    HAL_Delay(200);
    Gyro_ResetHeading();

    // PID state
    float integral = 0.0f;
    float last_error = 0.0f;

    // Tune these if needed
    const float ARC_KP = 70.0f;
    const float ARC_KI = 0.0f;
    const float ARC_KD = 0.0f;

    uint32_t last_time = HAL_GetTick();
    uint32_t last_oled = last_time;

    while (1)
    {
        uint32_t now = HAL_GetTick();

        float dt = (now - last_time) / 1000.0f;
        if (dt <= 0.0f)
            dt = 0.001f;
        last_time = now;

        Gyro_UpdateFromIMU(dt);
        float heading = Gyro_GetHeadingDeg();

        float error = target_heading - heading;
        float remaining = fabsf(error);

        // Stop near target
        if (remaining <= 1.0f)
            break;

        int pwm = base_pwm;

        // slow down near end
        if      (remaining > 20.0f) pwm = base_pwm;
        else if (remaining > 10.0f) pwm = (int)(base_pwm * 0.55f);
        else if (remaining >  5.0f) pwm = (int)(base_pwm * 0.45f);
        else                        pwm = (int)(base_pwm * 0.40f);

        if (pwm < 280)
            pwm = 280;

        // PID on heading error
        integral += error * dt;
        integral = clampf(integral, -30.0f, 30.0f);

        float derivative = (error - last_error) / dt;
        last_error = error;

        int corr = (int)(ARC_KP * error +
                         ARC_KI * integral +
                         ARC_KD * derivative);

        // Limit correction so it doesn't overpower the arc
        int corr_limit = (int)(0.20f * pwm);
        if (corr_limit < 50)
            corr_limit = 50;
        corr = clampi(corr, -corr_limit, corr_limit);
        int left_base;
        int right_base;

        if (steer_deg > 0)
               {
                   right_base = (int)(-pwm * 0.90f);
                   left_base = (int) (-pwm * 1.4);
               }
               else
               {
                   left_base = (int)(-pwm * 1.2f);
                   right_base = (int) (-pwm * 1.0f);
               }

        // Apply PID correction in reverse
        int left_cmd  = left_base  - corr;
        int right_cmd = right_base + corr;

        left_cmd  = clampi(left_cmd,  -pwmMax, pwmMax);
        right_cmd = clampi(right_cmd, -pwmMax, pwmMax);

        set_left_motor(left_cmd);
        set_right_motor(right_cmd);

        if (now - last_oled >= 100)
        {
            last_oled = now;

            OLED_Clear();

            OLED_ShowString(0, 0, (const uint8_t *)"Arc BW PID");

            snprintf(buf, sizeof(buf), "H:%.2f", heading);
            OLED_ShowString(0, 12, (uint8_t *)buf);

            snprintf(buf, sizeof(buf), "T:%.2f", target_heading);
            OLED_ShowString(0, 24, (uint8_t *)buf);

            snprintf(buf, sizeof(buf), "E:%.2f", error);
            OLED_ShowString(0, 36, (uint8_t *)buf);

            snprintf(buf, sizeof(buf), "C:%d", corr);
            OLED_ShowString(0, 48, (uint8_t *)buf);

            OLED_Refresh_Gram();
        }

        HAL_Delay(10);
    }

    Motor_stop();
    HAL_Delay(100);

    // Return steering to center
    Steering_ToUS(STEER_CENTER);
    HAL_Delay(150);
}


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
static int pwm_from_speed_cmps(float speed_cmps) {
  static const float speed_pts[] = {0.0f, 14.06f, 31.21f, 49.35f, 66.96f, 84.66f};
  static const int   pwm_pts[]   = {0, 1000, 2000, 3000, 4000, 5000};
  const int n = (int)(sizeof(speed_pts) / sizeof(speed_pts[0]));

  if (speed_cmps <= 0.0f)             return 0;
  if (speed_cmps >= speed_pts[n - 1]) return pwm_pts[n - 1];

  for (int i = 0; i < (n - 1); i++) {
    if (speed_cmps <= speed_pts[i + 1]) {
      float t = (speed_cmps - speed_pts[i]) / (speed_pts[i + 1] - speed_pts[i]);
      float pwm_f = (float)pwm_pts[i] + t * (float)(pwm_pts[i + 1] - pwm_pts[i]);
      return clampi((int)lroundf(pwm_f), 0, 5000);
    }
  }

  return pwm_pts[n - 1];
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
            //New_Drive_Straight_ToCM_USONIC((float)cm, 2200);
            Drive_Straight_ToCM((float)cm);
            DONE("SF done,cm=%.1f", cm_travelled_signed());
            return;
        }

	if (strncmp(t, "SB", 2) == 0) {
		int cm = atoi(t + 2);
		ACK("SB%03d", cm);
		Steering_ToUS(0);
		Drive_Straight_ToCM(-(float)cm);
		DONE("SB done,cm=%.1f", cm_travelled_signed());

		return;
	}

    // 2) In-place turns
    if (strncmp(t, "TL--", 4) == 0) {
        ACK("TL--");
        Steering_ToUS(0);
        Drive_Straight_ToCM(-10);
        HAL_Delay(100);
        Drive_Turn_Angle(35.0f, 0);
        //Drive_Turn_Angle(90.0f, +30, 2200);
        HAL_Delay(100);
        Drive_Straight_ToCM(25);
        HAL_Delay(100);
        Drive_Turn_Angle(-65.0f, 0);
        HAL_Delay(100);
        //works
        Drive_Straight_ToCM(40);
        HAL_Delay(100);
        Drive_Turn_Angle(-30.0f, 0);
        HAL_Delay(100);
        Steering_ToUS(0);
        GyroStraight_LockHeading();
        HAL_Delay(100);
        DriveForwardUntilObstacle(20, 3000);
        UART3_ACK("PICTURE");
        return;
    }
    if (strncmp(t, "TR--", 4) == 0) {
        ACK("TR--");
        Steering_ToUS(0);
        Drive_Straight_ToCM(-10);
        HAL_Delay(100);
        Drive_Turn_Angle(35.0f, 0);
        //Drive_Turn_Angle(90.0f, +30, 2200);
        HAL_Delay(100);
        Drive_Straight_ToCM(25);
        HAL_Delay(100);
        Drive_Turn_Angle(-65.0f, 0);
        HAL_Delay(100);
        //works
        Drive_Straight_ToCM(40);
        HAL_Delay(100);
        //HAL_Delay(100);
        //Drive_Turn_AngleBW(90.0f, 30, 2200);
        //HAL_Delay(100);
        //Drive_Straight_ToCM(70, 2200);
        //HAL_Delay(100);
        //Drive_Turn_Angle(180.0f, -30, 2200);
        //HAL_Delay(100);
        //Drive_Turn_AngleBW(45.0f, -30, 2200);
        //Drive_Straight_ToCM(10, 2200);
        //HAL_Delay(100);
        Drive_Turn_Angle(-30.0f, 0);
        HAL_Delay(100);
        Steering_ToUS(0);
        GyroStraight_LockHeading();
        HAL_Delay(100);
        DriveForwardUntilObstacle(20, 2000);
        UART3_ACK("PICTURE");
        return;
    }
    if (strncmp(t, "STR-", 4) == 0) {
            ACK("STR-");
            Drive_Straight_ToCM(-20);
            HAL_Delay(100);
            Drive_Turn_Angle(45.0f, 0);
            HAL_Delay(100);
            Drive_Turn_Angle(-50.0f, 0);
            HAL_Delay(100);
            Drive_Forward_Until_LeftIR_Then_LeftTurn(2000,100,3,40,90,-30);
            HAL_Delay(100);
            Drive_Straight_ToCM(10);
            HAL_Delay(100);
            Drive_Forward_Until_LeftIR_Then_LeftTurn(2000,100,3,40,90,-30);
            HAL_Delay(100);
            Drive_Forward_Until_LeftIR_Then_LeftTurn(2000,100,3,40,90,-30);
            HAL_Delay(100);
            Drive_Straight_ToCM(20);
            HAL_Delay(100);
            Drive_Forward_Until_LeftIR_detects(3000,100,3,40);
            HAL_Delay(100);
            Drive_Straight_ToCM(25);
            HAL_Delay(100);
            Drive_Turn_Angle(-90.0f, 0);
            HAL_Delay(100);
            Drive_Forward_Until_LeftIR_detects(3000,50,3,40);
            HAL_Delay(100);
            Drive_Straight_ToCM(-20);
            HAL_Delay(100);
            Drive_Turn_Angle(90.0f, 0);
            HAL_Delay(100);
            DriveForwardUntilObstacle(20, 3000);
            HAL_Delay(100);
            //Drive_Straight_ToCM(20, 1000);
            //Drive_Forward_Until_LeftIR_detects(1000,100,3,70);
            return;
    }
    if (strncmp(t, "STL-", 4) == 0) {
            ACK("STL-");
            Drive_Straight_ToCM(-10);
            HAL_Delay(100);
            Drive_Turn_Angle(-45.0f, 0);
            HAL_Delay(100);
            Drive_Turn_Angle(50.0f, 1);
            HAL_Delay(100);
            //Drive_Forward_Until_RightIR_Then_RightTurn(2000,100,3,60,90,+30);
            Drive_Straight_ToCM(50);
            Drive_Turn_Angle(90.0f, 0);
            HAL_Delay(100);
            Drive_Straight_ToCM(10);
            HAL_Delay(100);
            Drive_Turn_Angle(90.0f, 0);
            //Drive_Forward_Until_RightIR_Then_RightTurn(2000,100,3,60,90,+30);
            HAL_Delay(100);
            Drive_Straight_ToCM(100);
            //Drive_Forward_Until_RightIR_Then_RightTurn(2000,100,3,60,90,+30);
            HAL_Delay(100);
            Drive_Turn_Angle(90.0f,0);
            HAL_Delay(100);
            Drive_Straight_ToCM(10);
            HAL_Delay(100);
            Drive_Turn_Angle(90.0f,0);
            HAL_Delay(100);
            Drive_Straight_ToCM(80);
            //Drive_Forward_Until_RightIR_detects(3000,100,3,40);
            HAL_Delay(100);
            //Drive_Straight_ToCM(25, 3000);
            HAL_Delay(100);
            Drive_Turn_Angle(-90.0f, 0);
            HAL_Delay(100);
            Drive_Forward_Until_LeftIR_detects(3000,50,3,40);
            HAL_Delay(100);
            Drive_Straight_ToCM(-20);
            HAL_Delay(100);
            Drive_Turn_Angle(90.0f, 0);
            HAL_Delay(100);
            DriveForwardUntilObstacle(20, 3000);
            HAL_Delay(100);
            return;
    }
    if (strncmp(t, "TR++", 4) == 0) {
        ACK("TR+");
        Drive_Arc_Turn(90.0f, 30, 1900);
        DONE("TR done,deg=%.1f", Gyro_GetHeadingDeg());
        return;
    }
    if (strncmp(t, "TL++", 4) == 0) {
        ACK("TL++");
        Drive_Arc_Turn(90.0f, -29, 1900);
        DONE("TR done,deg=%.1f", Gyro_GetHeadingDeg());
        return;
    }

    if (strncmp(t, "TL180", 4) == 0) {
        ACK("TL180");
        Drive_Arc_Turn(180.0f, -29, 1900);
        DONE("TR done,deg=%.1f", Gyro_GetHeadingDeg());
        return;
    }

    if (strncmp(t, "TR180", 4) == 0) {
        ACK("TR180");
        Drive_Arc_Turn(1800.0f, 30, 1900);
        DONE("TR done,deg=%.1f", Gyro_GetHeadingDeg());
        return;
    }

    if (strncmp(t, "SNAP", 4) == 0) {
        Motor_stop();
        ACK("SNAP");
        DONE("SNAP done");
        return;
    }

    // 4) Stop / End
    if (strncmp(t, "STOP", 4) == 0) {
        Motor_stop();
        ACK("STOP");
        DONE("stop");
        return;
    }
    if (strncmp(t, "FIN", 3) == 0) {
        Motor_stop();
        ACK("FIN");
        DONE("end sequence");
        return;
    }

    // Unknown command
    ERR("unknown,%s", t);
}
 void Run_Command_As_RPi(const char *cmd_text)
{
    char cmd[32];
    snprintf(cmd, sizeof(cmd), "%s", cmd_text);
    process_command(cmd);
}

static void Test_Command_Script(void)
{
	Run_Command_As_RPi("SF100");
    HAL_Delay(500);

    Run_Command_As_RPi("SB020");
    HAL_Delay(500);

    Run_Command_As_RPi("TL--");
    HAL_Delay(500);

    Run_Command_As_RPi("TR--");
    HAL_Delay(500);

    Run_Command_As_RPi("STR-");
    HAL_Delay(500);

    Run_Command_As_RPi("STL-");
    HAL_Delay(500);

    Run_Command_As_RPi("BTR--");
    HAL_Delay(500);

    Run_Command_As_RPi("BTL--");
    HAL_Delay(500);

    Run_Command_As_RPi("STOP");
}

/* ---------------------- Minimal UART printf ----------------------- */
int _write(int file, char *ptr, int len) {
  HAL_UART_Transmit(&huart3, (uint8_t *)ptr, len, HAL_MAX_DELAY);
  return len;
}

// ------ new codes for task 2 ---------
void Motor_forward_PID(int PWM)
{
    // -----------------------------
    // Persistent state
    // -----------------------------
    static uint8_t initialized = 0;

    static float prev_left_enc = 0.0f;
    static float prev_right_enc = 0.0f;

    static float l_v_integral = 0.0f;
    static float r_v_integral = 0.0f;
    static float a_integral   = 0.0f;

    static float last_l_v_error = 0.0f;
    static float last_r_v_error = 0.0f;
    static float last_a_error   = 0.0f;

    static float v_ref_cm_s = 0.0f;

    static uint32_t last_time = 0;
    static uint32_t last_oled = 0;

    // -----------------------------
    // Constants
    // -----------------------------
    const float ACC_CM_S2      = 40.0f;
    const float MAX_SPEED_CM_S = 80.0f;

    const float FWD_KP         = 0.18f;
    const float FWD_KI         = 0.03f;
    const float FWD_KD         = 0.0f;
    const float FWD_I_CLAMP    = 4000.0f;
    const float FWD_CORR_CLAMP = 1200.0f;

    const float HEADING_KP = 1500.0f;
    const float HEADING_KI = 100.0f;
    const float HEADING_KD = 0.0f;

    char oled_line[32];

    // -----------------------------
    // Stop + reset on PWM == 0
    // -----------------------------
    if (PWM == 0)
    {
        Motor_stop();

        initialized = 0;

        prev_left_enc = 0.0f;
        prev_right_enc = 0.0f;

        l_v_integral = 0.0f;
        r_v_integral = 0.0f;
        a_integral = 0.0f;

        last_l_v_error = 0.0f;
        last_r_v_error = 0.0f;
        last_a_error = 0.0f;

        v_ref_cm_s = 0.0f;

        last_time = 0;
        last_oled = 0;
        return;
    }

    // -----------------------------
    // First-call initialization
    // -----------------------------
    if (!initialized)
    {
        reset_encoders();

        Steering_ToUS(0);
        HAL_Delay(50);

        Gyro_ResetHeading();
        HAL_Delay(20);
        Gyro_ResetHeading();

        prev_left_enc  = left_ticks_signed();
        prev_right_enc = right_ticks_signed();

        l_v_integral = 0.0f;
        r_v_integral = 0.0f;
        a_integral   = 0.0f;

        last_l_v_error = 0.0f;
        last_r_v_error = 0.0f;
        last_a_error   = 0.0f;

        v_ref_cm_s = 0.0f;

        last_time = HAL_GetTick();
        last_oled = last_time;

        initialized = 1;
    }

    // -----------------------------
    // Time step
    // -----------------------------
    uint32_t now = HAL_GetTick();
    float dt = (now - last_time) / 1000.0f;
    if (dt <= 0.0f) dt = 0.001f;
    last_time = now;

    // -----------------------------
    // Input shaping
    // -----------------------------
    int dir = (PWM >= 0) ? +1 : -1;
    int pwm_target = clampi(abs(PWM), 0, pwmMax);

    float target_speed_cm_s =
        ((float)pwm_target / (float)pwmMax) * MAX_SPEED_CM_S;

    if (v_ref_cm_s < target_speed_cm_s)
    {
        v_ref_cm_s += ACC_CM_S2 * dt;
        if (v_ref_cm_s > target_speed_cm_s)
            v_ref_cm_s = target_speed_cm_s;
    }
    else if (v_ref_cm_s > target_speed_cm_s)
    {
        v_ref_cm_s -= ACC_CM_S2 * dt;
        if (v_ref_cm_s < target_speed_cm_s)
            v_ref_cm_s = target_speed_cm_s;
    }

    // -----------------------------
    // Gyro update
    // -----------------------------
    Gyro_UpdateFromIMU(dt);
    float heading = Gyro_GetHeadingDeg();

    // -----------------------------
    // Encoder feedback
    // -----------------------------
    float left_enc  = left_ticks_signed();
    float right_enc = right_ticks_signed();

    float meas_l_enc_s = (left_enc  - prev_left_enc)  / dt;
    float meas_r_enc_s = (right_enc - prev_right_enc) / dt;

    prev_left_enc  = left_enc;
    prev_right_enc = right_enc;

    // -----------------------------
    // Wheel speed PID
    // -----------------------------
    float target_l_enc_s = (float)dir * v_ref_cm_s * COUNTS_PER_CM_L;
    float target_r_enc_s = (float)dir * v_ref_cm_s * COUNTS_PER_CM_R;

    float l_v_error = target_l_enc_s - meas_l_enc_s;
    l_v_integral += l_v_error * dt;
    l_v_integral = clampf(l_v_integral, -FWD_I_CLAMP, FWD_I_CLAMP);
    float l_v_derivative = (l_v_error - last_l_v_error) / dt;
    last_l_v_error = l_v_error;

    float corr_l = (FWD_KP * l_v_error) +
                   (FWD_KI * l_v_integral) +
                   (FWD_KD * l_v_derivative);
    corr_l = clampf(corr_l, -FWD_CORR_CLAMP, FWD_CORR_CLAMP);

    float r_v_error = target_r_enc_s - meas_r_enc_s;
    r_v_integral += r_v_error * dt;
    r_v_integral = clampf(r_v_integral, -FWD_I_CLAMP, FWD_I_CLAMP);
    float r_v_derivative = (r_v_error - last_r_v_error) / dt;
    last_r_v_error = r_v_error;

    float corr_r = (FWD_KP * r_v_error) +
                   (FWD_KI * r_v_integral) +
                   (FWD_KD * r_v_derivative);
    corr_r = clampf(corr_r, -FWD_CORR_CLAMP, FWD_CORR_CLAMP);

    int pwm_l = clampi((int)lroundf((float)pwm_target + corr_l), 0, pwmMax);
    int pwm_r = clampi((int)lroundf((float)pwm_target + corr_r), 0, pwmMax);

    // -----------------------------
    // Heading PID
    // -----------------------------
    float a_error = -heading;
    a_integral += a_error * dt;
    float a_derivative = (a_error - last_a_error) / dt;
    last_a_error = a_error;

    int heading_correction = (int)(
        (HEADING_KP * a_error) +
        (HEADING_KI * a_integral) +
        (HEADING_KD * a_derivative)
    );

    // -----------------------------
    // Motor mixing
    // -----------------------------
    int left_cmd  = dir * pwm_l - heading_correction;
    int right_cmd = dir * pwm_r + heading_correction;

    left_cmd  = clampi(left_cmd,  -pwmMax, pwmMax);
    right_cmd = clampi(right_cmd, -pwmMax, pwmMax);

    set_left_motor(left_cmd);
    set_right_motor(right_cmd);

    // -----------------------------
    // OLED debug
    // -----------------------------
    if (now - last_oled >= 100u)
    {
        last_oled = now;

        snprintf(oled_line, sizeof(oled_line), "PWM:%d", dir * pwm_target);
        OLED_ShowString(0, 24, (uint8_t*)oled_line);

        snprintf(oled_line, sizeof(oled_line), "L:%.1f R:%.1f", meas_l_enc_s, meas_r_enc_s);
        OLED_ShowString(0, 36, (uint8_t*)oled_line);

        snprintf(oled_line, sizeof(oled_line), "Head:%.2f", heading);
        OLED_ShowString(0, 48, (uint8_t*)oled_line);

        OLED_Refresh_Gram();
    }
}



void Run_Test_Sequence(void)
{
    char cmd1[] = "SF060";
    char cmd2[] = "TR--";
    char cmd3[] = "STR-";

    process_command(cmd1);
    HAL_Delay(200);

    process_command(cmd2);
    HAL_Delay(200);

    process_command(cmd3);
}

//void test_left1_right2(void){
//
//	// full test hard code for turn left1, right2
//	  Drive_Straight_ToCM_try(150.0f);
//	  Drive_Arc_Turn(45.0f, -29, 1900);
//	  Drive_Arc_Turn(45.0f, 30, 1900);
//	  Drive_Arc_Turn(45.0f, 30, 1900);
//	  Drive_Arc_Turn(35.0f, -29, 1900);
//	  Drive_Straight_ToCM_try(150.0f);
//	  Drive_via_Ping(35.0f);
//	  Drive_Arc_Turn(90.0f, 30, 1900);
//	  Drive_Straight_ToCM_LeftIR_Exceed2(1);
//
//
//	  Drive_Arc_Turn(180.0f, -45, 1900);
//	  Drive_Straight_ToCM_LeftIR_Exceed2(1);
//
//}
//
//void test_right1_left2(void){
//
//	  //full test hard code for turn right1, left2
//	  Drive_Straight_ToCM_try(150.0f);
//	  Drive_Arc_Turn(45.0f, 30, 1900);
//	  Drive_Arc_Turn(45.0f, -29, 1900);
//	  Drive_Arc_Turn(45.0f, -29, 1900);
//	  Drive_Arc_Turn(45.0f, 30, 1900);
//	  Drive_Straight_ToCM_try(150.0f);
//	  Drive_via_Ping(35.0f);
//	  Drive_Arc_Turn(90.0f, -29, 1900);
//	  Drive_Straight_ToCM_RightIR_Exceed2(1);
//
//
//	  Drive_Arc_Turn(180.0f, 45, 1900);
//	  Drive_Straight_ToCM_RightIR_Exceed2(1);
//}
//
//void test_right1_right2(void){
//	 // full test hard code for turn right1, left2
//	  Drive_Straight_ToCM_try(150.0f);
//	  Drive_Arc_Turn(45.0f, 30, 1900);
//	  Drive_Arc_Turn(45.0f, -29, 1900);
//	  Drive_Arc_Turn(45.0f, -29, 1900);
//	  Drive_Arc_Turn(45.0f, 30, 1900);
//	  Drive_Straight_ToCM_try(150.0f);
//	  Drive_via_Ping(35.0f);
//	  Drive_Arc_Turn(90.0f, 30, 1900);
//	  Drive_Straight_ToCM_LeftIR_Exceed2(1);
//
//
//	  Drive_Arc_Turn(180.0f, -45, 1900);
//	  Drive_Straight_ToCM_LeftIR_Exceed2(1);
//}
// ----- github copilot codes ----------
//void test_task2_three_steps(void){
//
//    uint32_t d;
//    float dx, dy, home_dist_cm;
//    float desired_heading_deg, delta_deg, right_turn_deg;
//
//    robot_x = 0.0f;
//    robot_y = 0.0f;
//    robot_heading_deg = 0.0f;
//
//    OLED_Clear();
//    OLED_ShowString(0, 0, (uint8_t*)"STEP1: Front obj");
//    OLED_Refresh_Gram();
//    HAL_Delay(900);
//
//    // Existing function: drive forward until US <= 150cm
//    Drive_Straight_ToCM_try(150.0f);
//    d = HCSR04_Read();
//    if (!(d >= 60 && d <= 150)){
//        OLED_Clear();
//        OLED_ShowString(0, 0, (uint8_t*)"STEP1 out of range");
//        OLED_Refresh_Gram();
//        HAL_Delay(1200);
//        return;
//    }
//
//    // Right-side bypass with 45-degree turns, then recenter
//    Drive_Arc_Turn(45.0f, 30, 1900);
//    Drive_Arc_Turn(45.0f, -29, 1900);
//    Drive_Arc_Turn(45.0f, -29, 1900);
//    Drive_Arc_Turn(45.0f, 30, 1900);
//
//    OLED_Clear();
//    OLED_ShowString(0, 0, (uint8_t*)"STEP2: Wall front");
//    OLED_Refresh_Gram();
//    HAL_Delay(900);
//
//    // Existing function: drive forward until US <= 140cm
//    Drive_Straight_ToCM_try(150.0f);
//
//    // Right 90 (updated as requested)
//    Drive_Arc_Turn(90.0f, 30, 1900);
//
//    // Follow wall with existing IR routine, then go behind wall
//    Drive_Straight_ToCM_LeftIR_Exceed2(1);
//    Drive_Arc_Turn(180.0f, 30, 1900);
//
//    OLED_Clear();
//    OLED_ShowString(0, 0, (uint8_t*)"STEP3: Return home");
//    OLED_Refresh_Gram();
//    HAL_Delay(900);
//
//    // Continue by existing IR routine
//    Drive_Straight_ToCM_LeftIR_Exceed2(1);
//
//    // Return-to-start math inline (no new helper function)
//    dx = -robot_x;
//    dy = -robot_y;
//    home_dist_cm = sqrtf(dx * dx + dy * dy);
//
//    if (home_dist_cm > 3.0f){
//        desired_heading_deg = atan2f(dx, dy) * 180.0f / PI;
//        delta_deg = wrap180f(desired_heading_deg - robot_heading_deg);
//        right_turn_deg = (delta_deg <= 0.0f) ? (-delta_deg) : (360.0f - delta_deg);
//        if (right_turn_deg > 359.0f) right_turn_deg = 359.0f;
//
//        Drive_Arc_Turn(right_turn_deg, 30, 1900);
//        Drive_Straight_ToCM_try(home_dist_cm);
//    }
//
//    OLED_Clear();
//    OLED_ShowString(0, 0, (uint8_t*)"Mission complete");
//    OLED_Refresh_Gram();
//    HAL_Delay(1200);
//}

// ------ github copilot codes end --------///


void test_left1_left2(void){
  // full test hard code for turn right1, left2

  float vertical_distance = 0.0f; // forward distance travelled
  char oled_line[32];
  float horizontal_distance = 0.0f; // L/R distance travelled along last wall
  vertical_distance += Drive_Straight_ToCM_try(150.0f, 35);
  vertical_distance += Drive_via_Ping(35.0f);
  Drive_Arc_Turn(45.0f, -29, 2400);
  Drive_Arc_Turn(45.0f, 30, 2400);
  Drive_Arc_Turn(45.0f, 30, 2400);
  Drive_Arc_Turn(35.0f, -29, 2400);
  vertical_distance += (30 * 4);
  vertical_distance += Drive_Straight_ToCM_try(150.0f, 35);
  vertical_distance += Drive_via_Ping(35.0f);
  Drive_Arc_Turn(90.0f, -29, 2400);

  vertical_distance += (30 * 1.0f);
  Drive_Straight_ToCM_RightIR_Exceed2(1);


  Drive_Arc_Turn(180.0f, 45, 1900);

  vertical_distance += (60 * 1.0f);
  horizontal_distance = ((Drive_Straight_ToCM_RightIR_Exceed2(1)+28) /2);
  float hypotenuse = sqrtf((vertical_distance * vertical_distance) + (horizontal_distance * horizontal_distance));
  float turn_angle_deg = 0.0f;
  turn_angle_deg = 180 - (atan2f(vertical_distance, horizontal_distance) * (180.0f / PI));
  //turn_angle_deg =  180 - acosf(vertical_distance / hypotenuse) * (180.0f / PI);

  OLED_Clear();
  snprintf(oled_line, sizeof(oled_line), "Vertical:%.2f", vertical_distance);
  OLED_ShowString(0, 0, (uint8_t*)oled_line);
  snprintf(oled_line, sizeof(oled_line), "Horizontal:%.2f", horizontal_distance);
  OLED_ShowString(0, 12, (uint8_t*)oled_line);
  snprintf(oled_line, sizeof(oled_line), "Degree:%.2f", turn_angle_deg);
  OLED_ShowString(0, 24, (uint8_t*)oled_line);
  OLED_Refresh_Gram();
  HAL_Delay(10000);
  Drive_Arc_Turn(turn_angle_deg, 30, 2400);

  DONE("L1L2 tri=%.1f", turn_angle_deg);
  vertical_distance += Drive_Straight_ToCM_try(hypotenuse, 15);

}

static inline float ArcCmdWithLastError(float nominal_deg, int steer_deg, float last_err_deg)
{
    float cmd = nominal_deg + ((steer_deg < 0) ? last_err_deg : -last_err_deg);
    return clampf(cmd, 1.0f, 220.0f); // choose max you allow
}

static bool RPi_SnapAndWaitLR(int *dir_out, int snap_int)
{
    char line[32];
    uint32_t start = HAL_GetTick();
    if (snap_int == 1){
    	UART3_SendLine("SNAP1");
    }
    else if(snap_int == 2){
    	UART3_SendLine("SNAP2");
    }
    else{
    	while(1){
			// UART3_ReadLine(0) returns immediately, so poll in chunks
			uint32_t chunk_ms = 200u;

			if (!UART3_ReadLine(line, sizeof(line), chunk_ms)) {
				continue; // no full line yet
			}
			if (strcmp(line, "START") == 0) {
				return true;
			}
    	}
    }

    while (1)
    {

        // UART3_ReadLine(0) returns immediately, so poll in chunks
        uint32_t chunk_ms = 200u;

        if (!UART3_ReadLine(line, sizeof(line), chunk_ms)) {
            continue; // no full line yet
        }

        if (strcmp(line, "LEFT") == 0) {
            *dir_out = +1;
            return true;
        }
        if (strcmp(line, "RIGHT") == 0) {
            *dir_out = -1;
            return true;
        }

        // ignore unrelated lines
    }
}


void test_2(){
  // full test hard code for turn right1, left2
  int first_direction = 1; // 1 for left1, -1 for right1
  int second_direction = -1; // 1 for left2, -1 for right2
  float vertical_distance = 0.0f; // forward distance traveled
  float last_turn_error = 0.0f;
  float turn_cmd = 0.0f;
  char oled_line[32];
  float horizontal_distance = 0.0f; // L/R distance traveled along last wall
  float drive_till_see_dist = 0.0f;
  //tdl slowdown for IR exceed, add move distance at the end for IR see, Check straight PID.

  if (!RPi_SnapAndWaitLR(&first_direction, 0)) {
      ERR("start timeout");
  }

  vertical_distance += Drive_Straight_ToCM_try(150.0f, 35);
  vertical_distance += Drive_via_Ping(35.0f);

  first_direction = 0;

  if (!RPi_SnapAndWaitLR(&first_direction, 1)) {
      ERR("SNAP timeout");
  }

  if (first_direction == 1){

    /* Turning sequence for left
    global_angle = Drive_Arc_Turn(45.0f + global_angle, -29, 2400);
    global_angle = Drive_Arc_Turn(45.0f + global_angle, 30, 2400);
    global_angle = Drive_Arc_Turn(45.0f + global_angle, 30, 2400);
    global_angle = Drive_Arc_Turn(50.0f + global_angle, -29, 2400);
    */

    last_turn_error = Drive_Arc_Turn(45.0f, -29, 2400);

    turn_cmd = ArcCmdWithLastError(45.0f, 30, last_turn_error);
    last_turn_error = Drive_Arc_Turn(turn_cmd, 30, 2400);

    turn_cmd = ArcCmdWithLastError(45.0f, 30, last_turn_error);
    last_turn_error = Drive_Arc_Turn(turn_cmd, 30, 2400);

    turn_cmd = ArcCmdWithLastError(42.0f, -29, last_turn_error);
    last_turn_error = Drive_Arc_Turn(turn_cmd, -29, 2400);

  } else {
    /* turning seq for right
    last_turn_error = Drive_Arc_Turn(45.0f + last_turn_error, 30, 2400);
    last_turn_error = Drive_Arc_Turn(45.0f + last_turn_error, -29, 2400);
    last_turn_error = Drive_Arc_Turn(45.0f + last_turn_error, -29, 2400);
    last_turn_error = Drive_Arc_Turn(45.0f + last_turn_error, 30, 2400);
    */

    last_turn_error = Drive_Arc_Turn(45.0f, 30, 2400);

    turn_cmd = ArcCmdWithLastError(45.0f, -29, last_turn_error);
    last_turn_error = Drive_Arc_Turn(turn_cmd, -29, 2400);

    turn_cmd = ArcCmdWithLastError(45.0f, -29, last_turn_error);
    last_turn_error = Drive_Arc_Turn(turn_cmd, -29, 2400);

    turn_cmd = ArcCmdWithLastError(45.0f, 30, last_turn_error);
    last_turn_error = Drive_Arc_Turn(turn_cmd, 30, 2400);
  }
  vertical_distance += (30 * 4);
  vertical_distance += Drive_Straight_ToCM_try(150.0f, 35);
  vertical_distance += Drive_via_Ping(35.0f);



  if (!RPi_SnapAndWaitLR(&second_direction, 2)) {
      ERR("SNAP timeout");
  }

  if(second_direction == 1){
    turn_cmd = ArcCmdWithLastError(90.0f, -29, last_turn_error);
    //last_turn_error = Drive_Arc_Turn(90.0f + last_turn_error, -29, 2400);
    last_turn_error = Drive_Arc_Turn(turn_cmd, -29, 2400);

    Drive_Straight_ToCM_RightIR_Exceed2(1);
    Drive_Straight_ToCM_RightIR_See(-1);
    Drive_Straight_ToCM_try(10,10);

    turn_cmd = ArcCmdWithLastError(180.0f, 45, last_turn_error);
    //last_turn_error = Drive_Arc_Turn(180.0f + last_turn_error, 45, 1900);

    last_turn_error = Drive_Arc_Turn(turn_cmd, 45, 1900);
    Drive_Straight_ToCM_RightIR_Exceed2(-1);
    Drive_Straight_ToCM_RightIR_See(1);
    horizontal_distance += Drive_Straight_ToCM_RightIR_Exceed2(1);
    horizontal_distance -= Drive_Straight_ToCM_RightIR_See(-1);
    Drive_Straight_ToCM_try(10,10);
    horizontal_distance = horizontal_distance/2;
  } else {

    turn_cmd = ArcCmdWithLastError(90.0f, 30, last_turn_error);
	  //last_turn_error = Drive_Arc_Turn(90.0f + last_turn_error, 30, 2400);
    last_turn_error = Drive_Arc_Turn(turn_cmd, 30, 2400);

    Drive_Straight_ToCM_LeftIR_Exceed2(1);
    Drive_Straight_ToCM_LeftIR_See(-1);
    Drive_Straight_ToCM_try(10,10);

    turn_cmd = ArcCmdWithLastError(180.0f, -45, last_turn_error);
    //last_turn_error = Drive_Arc_Turn(180.0f + last_turn_error, -45, 1900);
    last_turn_error = Drive_Arc_Turn(turn_cmd, -45, 1900);

    Drive_Straight_ToCM_LeftIR_Exceed2(-1);
    Drive_Straight_ToCM_LeftIR_See(1);

    horizontal_distance += Drive_Straight_ToCM_LeftIR_Exceed2(1); //42 53
    horizontal_distance -= Drive_Straight_ToCM_LeftIR_See(-1);
    Drive_Straight_ToCM_try(10,10);
    horizontal_distance = horizontal_distance/2;
  }

  vertical_distance += (30 * 1.0f); // add turn radius for 90 deg turn
  vertical_distance += (60 * 1.0f); // add turn radius for 180 deg turn
  float hypotenuse = sqrtf((vertical_distance * vertical_distance) + (horizontal_distance * horizontal_distance));
  float turn_angle_deg = 0.0f;
  turn_angle_deg = 178 - (atan2f(vertical_distance, horizontal_distance) * (180.0f / PI));
  //turn_angle_deg =  180 - acosf(vertical_distance / hypotenuse) * (180.0f / PI);

  OLED_Clear();
  snprintf(oled_line, sizeof(oled_line), "Vertical:%.2f", vertical_distance);
  OLED_ShowString(0, 0, (uint8_t*)oled_line);
  snprintf(oled_line, sizeof(oled_line), "Horizontal:%.2f", horizontal_distance);
  OLED_ShowString(0, 12, (uint8_t*)oled_line);
  snprintf(oled_line, sizeof(oled_line), "Degree:%.2f", turn_angle_deg);
  OLED_ShowString(0, 24, (uint8_t*)oled_line);
  OLED_Refresh_Gram();
  HAL_Delay(300);
  //turn straight back
  if(second_direction == 1){
	  turn_cmd = ArcCmdWithLastError(turn_angle_deg, 30, last_turn_error);
	  Drive_Arc_Turn(turn_cmd, 30, 2400);
  }else{
	  turn_cmd = ArcCmdWithLastError(turn_angle_deg, -29, last_turn_error);
	  Drive_Arc_Turn(turn_cmd, -29, 2400);
  }
  //to turn 90, then back
  /*
  if(second_direction == 1){
	  turn_cmd = ArcCmdWithLastError(90, 30, last_turn_error);
	  Drive_Arc_Turn(turn_cmd, 30, 2400);
  }else{
	  turn_cmd = ArcCmdWithLastError(90, -29, last_turn_error);
	  Drive_Arc_Turn(turn_cmd, -29, 2400);
  }
  */

  HAL_Delay(100);
  //Drive_Straight_ToCM_try(vertical_distance*0.5, 20);
  /*
  if(second_direction == 1){
	  turn_cmd = ArcCmdWithLastError(20, 30, last_turn_error);
	  Drive_Arc_Turn(turn_cmd, 30, 2400);
  }else{
	  turn_cmd = ArcCmdWithLastError(20, -29, last_turn_error);
	  Drive_Arc_Turn(turn_cmd, -29, 2400);
  }
  */
  //Drive_Straight_ToCM_try((vertical_distance*0.5), 20);

  vertical_distance += Drive_Straight_ToCM_try((hypotenuse+50), 20);
  Drive_via_Ping(35.0f);
  if(second_direction == 1){
	  turn_cmd = ArcCmdWithLastError(90, 30, last_turn_error);
	  Drive_Arc_Turn(turn_cmd, 30, 2400);
	  turn_cmd = ArcCmdWithLastError(90, -29, last_turn_error);
	  Drive_Arc_Turn(turn_cmd, 30, 2400);
	  Drive_Straight_ToCM_try(100,20);
  }else{
	  turn_cmd = ArcCmdWithLastError(90, -29, last_turn_error);
	  Drive_Arc_Turn(turn_cmd, -29, 2400);
	  turn_cmd = ArcCmdWithLastError(90, 30, last_turn_error);
	  Drive_Arc_Turn(turn_cmd, -29, 2400);
	  Drive_Straight_ToCM_try(100,20);
  }

  DONE("DONE");
}

/* =============================  main_  ============================= */
int main(void)
{
  HAL_Init();
  HCSR04_InitDWT();
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
  MX_ADC2_Init();

  /* Peripherals start */
  HAL_TIM_Encoder_Start(&htim2, TIM_CHANNEL_ALL);  // Left encoder (TIM2)
  HAL_TIM_Encoder_Start(&htim5, TIM_CHANNEL_ALL);  // Right encoder (TIM5)
  MotorDrive_enable();                             // PWM for motors
  HAL_TIM_PWM_Start(&htim12, TIM_CHANNEL_2);       // Servo PWM

  OLED_Init();
  OLED_Clear();
  OLED_ShowString(0,0,(uint8_t*)"Task_20/3 835am code uploaded");

  HAL_Delay(100);
  if (ICM20948_Detect() == 0 && ICM20948_Init() == 0) {
    OLED_ShowString(0, 12, (uint8_t*)"IMU OK");
  } else {
    OLED_ShowString(0, 12, (uint8_t*)"IMU FAIL");
  }
  OLED_Refresh_Gram();

  HAL_Delay(1000);


  //test_2();
  Drive_Straight_ToCM_try(10000000.0f,50);
  //Drive_Straight_ToCM_RightIR_See(1);
  //Drive_Straight_ToCM_try(100.0f,15);
  //Drive_Straight_ToCM_try(-10.0f,15);

  //Drive_via_Ping(35.0f);
//  //test_right1_right2();
//test_left1_left2();
  //Drive_Straight_ToCM_LeftIR_Exceed2(1);
//  Drive_Arc_Turn(45.0f, 30, 2400);
//  Drive_Arc_Turn(45.0f, -29, 2400);
//  Drive_Arc_Turn(45.0f, -29, 2400);
//  Drive_Arc_Turn(45.0f, 30, 2400);
////  Drive_Arc_Turn(90.0f, 30, 1900);
////  Drive_Straight_ToCM_LeftIR_Exceed2(1);

  //Motor_forward_PID(3000);`
 //Drive_Forward_Until_LeftIR_Exceeds(3000,50,3,40);
//Drive_Straight_ToCM_LeftIR_Exceed(1);
//Drive_Straight_ToCM_RightIR_Exceed(1);
 //Drive_Straight_ToCM_LeftIR_Exceed(1);
/*

  Drive_Straight_ToCM_try(150.0f);
  //Drive_Forward_Until_LeftIR_Exceeds(3000,50,3,40);
  Drive_Arc_Turn(45.0f, -29, 1900);
  //HAL_Delay(1000);
  Drive_Arc_Turn(45.0f, +30, 1900);
  //HAL_Delay(1000);
  Drive_Arc_Turn(45.0f, +30, 1900);
  Drive_Straight_ToCM_try(10.0f);
  Drive_Arc_Turn(45.0f, -29, 1900);
  Drive_Straight_ToCM_try(150.0f);
  Drive_Arc_Turn(90.0f, 30, 1900);
*/

  //HAL_Delay(1000);
  //Drive_Straight_ToCM(150.0f);-
  //HAL_Delay(1000);
  //Run_Test_Sequence();


  while (1)
    {
        uint8_t ch;
        if (HAL_UART_Receive(&huart3, &ch, 1, HAL_MAX_DELAY) == HAL_OK)
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
  sConfig.Channel = ADC_CHANNEL_2;
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
  * @brief ADC2 Initialization Function
  * @param None
  * @retval None
  */
static void MX_ADC2_Init(void)
{

  /* USER CODE BEGIN ADC2_Init 0 */

  /* USER CODE END ADC2_Init 0 */

  ADC_ChannelConfTypeDef sConfig = {0};

  /* USER CODE BEGIN ADC2_Init 1 */

  /* USER CODE END ADC2_Init 1 */

  /** Configure the global features of the ADC (Clock, Resolution, Data Alignment and number of conversion)
  */
  hadc2.Instance = ADC2;
  hadc2.Init.ClockPrescaler = ADC_CLOCK_SYNC_PCLK_DIV2;
  hadc2.Init.Resolution = ADC_RESOLUTION_12B;
  hadc2.Init.ScanConvMode = DISABLE;
  hadc2.Init.ContinuousConvMode = DISABLE;
  hadc2.Init.DiscontinuousConvMode = DISABLE;
  hadc2.Init.ExternalTrigConvEdge = ADC_EXTERNALTRIGCONVEDGE_NONE;
  hadc2.Init.ExternalTrigConv = ADC_SOFTWARE_START;
  hadc2.Init.DataAlign = ADC_DATAALIGN_RIGHT;
  hadc2.Init.NbrOfConversion = 1;
  hadc2.Init.DMAContinuousRequests = DISABLE;
  hadc2.Init.EOCSelection = ADC_EOC_SINGLE_CONV;
  if (HAL_ADC_Init(&hadc2) != HAL_OK)
  {
    Error_Handler();
  }

  /** Configure for the selected ADC regular channel its corresponding rank in the sequencer and its sample time.
  */
  sConfig.Channel = ADC_CHANNEL_7;
  sConfig.Rank = 1;
  sConfig.SamplingTime = ADC_SAMPLETIME_3CYCLES;
  if (HAL_ADC_ConfigChannel(&hadc2, &sConfig) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN ADC2_Init 2 */

  /* USER CODE END ADC2_Init 2 */

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
  // IR Left  (PA2 → ADC1_IN2)
  // IR Right (PA3 → ADC2_IN3)


  __HAL_RCC_GPIOA_CLK_ENABLE();

  GPIO_InitStruct.Mode = GPIO_MODE_ANALOG;
  GPIO_InitStruct.Pull = GPIO_NOPULL;

  GPIO_InitStruct.Pin = GPIO_PIN_2;   // PA2
  HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

  GPIO_InitStruct.Pin = GPIO_PIN_3;   // PA3
  HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);
  /* USER CODE END MX_GPIO_Init_2 */


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
