"""
Animated Terminal Visualizer for Simulated Neckband LRAs
Lead Researcher: Tejaswi E. (24BCE5518)

Renders real-time visual tactile feedback in the console when physical
DRV2605L actuators are not connected.
"""

import sys
import time
from .haptic_encoder import HapticCommand


class TerminalHapticVisualizer:
    """Renders ANSI-colored visualizer for 3 neckband LRAs."""

    # ANSI Colors
    RESET = "\033[0m"
    BOLD = "\033[1m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    CYAN = "\033[96m"
    GRAY = "\033[90m"

    def __init__(self):
        self.last_render_time = 0.0

    def render(self, cmd: HapticCommand, head_yaw_deg: float, latency_ms: float):
        """Displays formatted neckband state line."""
        now = time.time()
        # Rate limit console refresh to avoid screen flicker
        if now - self.last_render_time < 0.1:
            return
        self.last_render_time = now

        # Format urgency indicator
        if cmd.urgency_level == "IMMINENT":
            urg_color = self.RED
            urg_symbol = "[!] IMMINENT"
        elif cmd.urgency_level == "APPROACHING":
            urg_color = self.YELLOW
            urg_symbol = "[~] APPROACHING"
        else:
            urg_color = self.GREEN
            urg_symbol = "[+] SAFE"

        # Construct visual nodes for (L) (C) (R) using ASCII-safe characters
        def get_node(channel_name: str) -> str:
            if cmd.motor_channel == channel_name and cmd.should_pulse:
                return f"{urg_color}[*PULSING*]{self.RESET}"
            return f"{self.GRAY}[  IDLE   ]{self.RESET}"

        left_node = get_node("LEFT")
        center_node = get_node("CENTER")
        right_node = get_node("RIGHT")

        line = (
            f"\r{self.BOLD}[HAPTIC NECKBAND]{self.RESET} "
            f"L: {left_node}  C: {center_node}  R: {right_node} | "
            f"Yaw: {self.CYAN}{head_yaw_deg:+5.1f} deg{self.RESET} | "
            f"Dist: {cmd.distance_m:4.2f}m | "
            f"TTC: {cmd.ttc_seconds:4.2f}s ({urg_symbol}{self.RESET}) | "
            f"Wave: {self.BOLD}{cmd.waveform_name}{self.RESET} (#{cmd.waveform_id}) | "
            f"Lag: {latency_ms:4.1f}ms"
        )

        try:
            sys.stdout.write(line)
            sys.stdout.flush()
        except UnicodeEncodeError:
            # Fallback plain encoding for constrained Windows consoles
            sys.stdout.write(line.encode("ascii", "replace").decode("ascii"))
            sys.stdout.flush()
