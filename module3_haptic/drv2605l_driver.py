"""
TI DRV2605L Haptic Motor Driver with PCA9548A I2C Mux Support
Lead Researcher: Tejaswi E. (24BCE5518)

Configures TI DRV2605L for Linear Resonant Actuators (LRA),
loads waveform sequence from ROM library, and fires directional pulses.

Hardware: 3× DRV2605L (all at 0x5A) behind PCA9548A I2C Multiplexer (0x70)
  - Mux Channel 0 → DRV2605L #1 (Left LRA)
  - Mux Channel 1 → DRV2605L #2 (Center LRA)
  - Mux Channel 2 → DRV2605L #3 (Right LRA)
"""

import time
from typing import Optional
from config import (
    DRV2605L_I2C_ADDR,
    PCA9548A_I2C_ADDR,
    I2C_BUS_ID,
    MOTOR_LEFT,
    MOTOR_CENTER,
    MOTOR_RIGHT,
    MUX_CHANNEL_MAP,
    SIMULATION_MODE,
)


class DRV2605LDriver:
    """I2C driver for TI DRV2605L haptic driver IC with PCA9548A mux routing."""

    # Registers
    REG_STATUS = 0x00
    REG_MODE = 0x01
    REG_RTPIN = 0x02
    REG_LIBRARY = 0x03
    REG_WAVESEQ1 = 0x04
    REG_WAVESEQ2 = 0x05
    REG_GO = 0x0C
    REG_FEEDBACK = 0x1A
    REG_CONTROL3 = 0x1D

    # Mode values
    MODE_INTTRIG = 0x00
    MODE_STANDBY = 0x40

    def __init__(self, i2c_bus: int = I2C_BUS_ID, address: int = DRV2605L_I2C_ADDR,
                 mux_address: int = PCA9548A_I2C_ADDR):
        self.bus_id = i2c_bus
        self.address = address
        self.mux_address = mux_address
        self.is_connected = False
        self.bus = None
        self._current_mux_channel = -1  # Track active channel to avoid redundant switches

        if not SIMULATION_MODE:
            self._init_hardware()

    def _select_mux_channel(self, channel: int):
        """
        Switch PCA9548A multiplexer to the specified channel (0-7).
        Writes a bitmask to the mux control register to enable only one channel.
        """
        if self.bus is None:
            return False

        if channel == self._current_mux_channel:
            return True  # Already on the right channel

        try:
            # PCA9548A: write a byte where bit N enables channel N
            self.bus.write_byte(self.mux_address, 1 << channel)
            self._current_mux_channel = channel
            return True
        except Exception as e:
            print(f"[DRV2605L] Mux channel select error (ch={channel}): {e}")
            return False

    def _init_hardware(self):
        """
        Configure all 3× DRV2605L drivers for LRA operation via PCA9548A mux.
        Iterates through each mux channel and initializes the DRV2605L behind it.
        """
        try:
            import smbus2
            self.bus = smbus2.SMBus(self.bus_id)

            init_count = 0
            for motor, channel in MUX_CHANNEL_MAP.items():
                try:
                    self._select_mux_channel(channel)
                    # Set mode to internal trigger
                    self.bus.write_byte_data(self.address, self.REG_MODE, self.MODE_INTTRIG)
                    # Select Library 6 (LRA Library)
                    self.bus.write_byte_data(self.address, self.REG_LIBRARY, 0x06)
                    # Configure feedback control for LRA (set LRA mode bit 7 in REG_FEEDBACK)
                    self.bus.write_byte_data(self.address, self.REG_FEEDBACK, 0xB6)
                    init_count += 1
                    print(f"[DRV2605L] Initialized {motor} motor via mux channel {channel}")
                except Exception as e:
                    print(f"[DRV2605L] Init failed for {motor} (mux ch={channel}): {e}")

            self.is_connected = init_count > 0
            print(f"[DRV2605L] {init_count}/3 haptic drivers initialized via PCA9548A mux (0x{self.mux_address:02X})")

        except Exception as e:
            print(f"[DRV2605L] Hardware init bypassed ({e}). Operating in simulation mode.")
            self.is_connected = False

    def trigger_waveform(self, motor_channel: str, waveform_id: int):
        """
        Loads specified ROM waveform into slot 1 and sets GO bit.
        Routes actuation to specified neckband channel (LEFT, CENTER, RIGHT)
        by first switching the PCA9548A mux to the correct channel.
        """
        if self.is_connected and self.bus is not None:
            try:
                # Route to the correct DRV2605L via mux
                mux_ch = MUX_CHANNEL_MAP.get(motor_channel)
                if mux_ch is None:
                    print(f"[DRV2605L] Unknown motor channel: {motor_channel}")
                    return False

                if not self._select_mux_channel(mux_ch):
                    return False

                # Slot 1: target waveform
                self.bus.write_byte_data(self.address, self.REG_WAVESEQ1, int(waveform_id) & 0x7F)
                # Slot 2: 0x00 to terminate sequence
                self.bus.write_byte_data(self.address, self.REG_WAVESEQ2, 0x00)
                # Set GO bit (0x01)
                self.bus.write_byte_data(self.address, self.REG_GO, 0x01)
                return True
            except Exception as e:
                print(f"[DRV2605L] I2C write error ({motor_channel}): {e}")
                return False

        # In simulation mode, actuation is confirmed virtually
        return True

    def stop(self):
        """Place all drivers into standby."""
        if self.is_connected and self.bus is not None:
            for motor, channel in MUX_CHANNEL_MAP.items():
                try:
                    self._select_mux_channel(channel)
                    self.bus.write_byte_data(self.address, self.REG_MODE, self.MODE_STANDBY)
                except Exception:
                    pass

    def close(self):
        self.stop()
        if self.bus is not None:
            try:
                self.bus.close()
            except Exception:
                pass
            self.bus = None
