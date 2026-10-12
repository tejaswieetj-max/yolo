"""
InvenSense MPU6050 6-Axis IMU Driver & Kinematic Filter
Lead Researcher: Andrea A. (24BCE1142)

Samples accelerometer and gyroscope at 100 Hz over I2C (Address 0x68).
Applies a Complementary Filter (alpha = 0.98) to derive drift-free head yaw and pitch.
"""

import time
import math
from typing import Tuple
from config import (
    MPU6050_I2C_ADDR,
    I2C_BUS_ID,
    COMPLEMENTARY_FILTER_ALPHA,
    SIMULATION_MODE,
)


class MPU6050Driver:
    """I2C driver and Complementary Filter for MPU6050 6-DOF IMU."""

    # Register addresses
    PWR_MGMT_1 = 0x6B
    SMPLRT_DIV = 0x19
    CONFIG = 0x1A
    GYRO_CONFIG = 0x1B
    ACCEL_CONFIG = 0x1C
    ACCEL_XOUT_H = 0x3B
    GYRO_XOUT_H = 0x43

    # Scale factors for +/- 250 deg/s and +/- 2g
    GYRO_SCALE = 131.0       # LSB / (deg/s)
    ACCEL_SCALE = 16384.0    # LSB / g

    def __init__(self, i2c_bus: int = I2C_BUS_ID, address: int = MPU6050_I2C_ADDR):
        self.bus_id = i2c_bus
        self.address = address
        self.is_connected = False
        self.bus = None

        # Filter state variables
        self.yaw_deg: float = 0.0
        self.pitch_deg: float = 0.0
        self.roll_deg: float = 0.0
        self.last_update_time: float = time.time()

        if not SIMULATION_MODE:
            self._init_hardware()

    def _init_hardware(self):
        """Wake up MPU6050 and set clock source to PLL with X axis gyroscope reference."""
        try:
            import smbus2
            self.bus = smbus2.SMBus(self.bus_id)
            # Wake up device (clear sleep bit in PWR_MGMT_1)
            self.bus.write_byte_data(self.address, self.PWR_MGMT_1, 0x01)
            # Sample rate divider
            self.bus.write_byte_data(self.address, self.SMPLRT_DIV, 0x07)
            # Gyro full scale range +/- 250 deg/s
            self.bus.write_byte_data(self.address, self.GYRO_CONFIG, 0x00)
            # Accel full scale range +/- 2g
            self.bus.write_byte_data(self.address, self.ACCEL_CONFIG, 0x00)
            self.is_connected = True
            print(f"[MPU6050] Successfully initialized at I2C address 0x{self.address:02X}")
        except Exception as e:
            print(f"[MPU6050] Hardware init failed ({e}). Falling back to simulation mode.")
            self.is_connected = False

    def _read_word_2c(self, reg: int) -> int:
        """Read signed 16-bit word from two consecutive registers."""
        high = self.bus.read_byte_data(self.address, reg)
        low = self.bus.read_byte_data(self.address, reg + 1)
        val = (high << 8) + low
        if val >= 0x8000:
            return -((65535 - val) + 1)
        return val

    def read_raw_sensors(self) -> Tuple[float, float, float, float, float, float]:
        """
        Returns (ax_g, ay_g, az_g, gx_dps, gy_dps, gz_dps).
        """
        if self.is_connected and self.bus is not None:
            try:
                ax = self._read_word_2c(self.ACCEL_XOUT_H) / self.ACCEL_SCALE
                ay = self._read_word_2c(self.ACCEL_XOUT_H + 2) / self.ACCEL_SCALE
                az = self._read_word_2c(self.ACCEL_XOUT_H + 4) / self.ACCEL_SCALE

                gx = self._read_word_2c(self.GYRO_XOUT_H) / self.GYRO_SCALE
                gy = self._read_word_2c(self.GYRO_XOUT_H + 2) / self.GYRO_SCALE
                gz = self._read_word_2c(self.GYRO_XOUT_H + 4) / self.GYRO_SCALE
                return ax, ay, az, gx, gy, gz
            except Exception as e:
                print(f"[MPU6050] Read error: {e}")

        return 0.0, 0.0, 1.0, 0.0, 0.0, 0.0

    def update_orientation(self) -> Tuple[float, float, float]:
        """
        Executes Complementary Filter step:
        Returns current drift-compensated (yaw_deg, pitch_deg, roll_deg).
        """
        now = time.time()
        dt = max(now - self.last_update_time, 1e-4)
        self.last_update_time = now

        ax, ay, az, gx, gy, gz = self.read_raw_sensors()

        # Compute pitch and roll from accelerometer gravity vector
        accel_pitch_deg = math.degrees(math.atan2(ax, math.sqrt(ay * ay + az * az)))
        accel_roll_deg = math.degrees(math.atan2(ay, az))

        # Complementary Filter:
        # High-pass filter on integrated gyro angular rate + Low-pass on accelerometer tilt
        alpha = COMPLEMENTARY_FILTER_ALPHA
        self.pitch_deg = alpha * (self.pitch_deg + gy * dt) + (1.0 - alpha) * accel_pitch_deg
        self.roll_deg = alpha * (self.roll_deg + gx * dt) + (1.0 - alpha) * accel_roll_deg

        # Yaw is integrated from Z-gyro (gyroscope z-axis gives rate of head turn)
        self.yaw_deg = self.yaw_deg + gz * dt
        # Normalize yaw to [-180, 180]
        if self.yaw_deg > 180.0:
            self.yaw_deg -= 360.0
        elif self.yaw_deg < -180.0:
            self.yaw_deg += 360.0

        return self.yaw_deg, self.pitch_deg, self.roll_deg

    def reset_yaw(self):
        """Zero the yaw reference angle."""
        self.yaw_deg = 0.0

    def close(self):
        if self.bus is not None:
            try:
                self.bus.close()
            except Exception:
                pass
            self.bus = None
