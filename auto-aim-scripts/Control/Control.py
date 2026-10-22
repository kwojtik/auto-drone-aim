import logging
import struct
import threading
import time
from collections import deque

import serial

import Control.msp_helper as msp

log = logging.getLogger(__name__)

MSP_REQUEST_HEADER = b'$M<'

# MSP_SET_RAW_RC carries raw RX channels, which the FC remaps with `map` (CLI) - AETR1234
ROLL, PITCH, THROTTLE, YAW = 0, 1, 2, 3
# MSP_RC returns channels already remapped, in the FC's internal order (Betaflight / INAV):
# roll, pitch, yaw, throttle. AUX channels have the same index in both orders.
MSP_RC_THROTTLE = 3
CHANNEL_COUNT = 8

RC_MIN = 1000
RC_MID = 1500
RC_MAX = 2000

# Values outside this range from MSP_RC mean no valid RX data (e.g. failsafe / not bound)
RC_VALID_MIN = 885
RC_VALID_MAX = 2115

MIN_SEND_RATE_HZ = 50
# Consecutive frames that must be sent before the stream is reported as ready
STREAM_READY_FRAMES = 25
# Sent throttle values remembered to recognise MSP_RC echoing them back (~0.5 s at 50 Hz)
SENT_THROTTLE_HISTORY = 25
# Consecutive invalid MSP_RC readings before it is logged (single missed replies are normal)
RC_INVALID_LOG_FRAMES = 25


def clamp(value, low, high):
    return max(low, min(high, value))


class Control:
    def __init__(self, companion_computer="/dev/ttyACM0", baud_rate=115200,
                 send_rate_hz=50, target_timeout=0.3, max_target_timeout=1.0,
                 max_offset=200, max_throttle_offset=25, throttle_trim=50,
                 throttle_i_gain=50, max_throttle_integral=150,
                 target_area=0.1, deadband=0.05):
        # Gains are stick deflection (us) for a full normalised error (1.0)
        self.yaw_gain = 200
        self.pitch_gain = 150
        self.throttle_gain = 100
        self.max_offset = max_offset            # max pitch / yaw deflection from centre
        self.max_throttle_offset = max_throttle_offset
        self.target_area = target_area          # box area / frame area to hold the target at
        self.deadband = deadband                # normalised error ignored around zero
        self.send_period = 1.0 / max(send_rate_hz, MIN_SEND_RATE_HZ)
        # Seconds without target -> sticks centred. Grows up to `max_target_timeout`
        # when inference is slower, so commands do not drop out between detections.
        self.target_timeout = target_timeout
        self.max_target_timeout = max_target_timeout
        self.throttle_trim = throttle_trim      # subtracted from the pilot's throttle while overridden
        # The stick position at activation is rarely the hover throttle, and the fixed trim
        # plus the small proportional term cannot always stop a climb. The integral of the
        # vertical error learns the remaining hover correction (us per second per unit error)
        # and is kept when the target is lost, so the drone does not climb away again.
        self.throttle_i_gain = throttle_i_gain
        self.max_throttle_integral = max_throttle_integral

        self.serial_port = None
        self._lock = threading.Lock()
        self._offsets = (0, 0, 0)               # pitch, yaw, throttle
        self._throttle_integral = 0.0
        self._overridden = False                # MSP_RC echoes our throttle -> override active
        self._last_target_time = 0.0
        self._detection_interval = None         # smoothed time between detections
        self._last_rc = None
        self._base_throttle = None              # pilot's throttle from the RX (before override)
        self._sent_throttles = deque(maxlen=SENT_THROTTLE_HISTORY)
        self._running = False
        self._thread = None
        self._streaming = threading.Event()     # set once MSP_SET_RAW_RC is streamed steadily
        self._rc_valid = None                   # last MSP_RC validity, to log only changes

        try:
            self.serial_port = serial.Serial(companion_computer, baud_rate, timeout=0.002)
        except serial.SerialException as e:
            log.error(f"Error connecting to serial port {companion_computer}: {e}")
            return

        log.info(f"Connected to {companion_computer} @ {baud_rate}, FC variant: {self.get_FC_variant()}")

        # Frames are streamed all the time, so when MSP OVERRIDE is switched on
        # the FC already has fresh values instead of stale or default ones.
        # Nothing is sent until MSP_RC has returned valid values for all channels.
        self._running = True
        self._thread = threading.Thread(target=self._send_loop, daemon=True)
        self._thread.start()

    def is_connected(self):
        return self.serial_port is not None and self.serial_port.is_open

    def is_streaming(self):
        return self._streaming.is_set()

    def wait_until_streaming(self, timeout=None):
        """Blocks until MSP_SET_RAW_RC frames with valid values for all channels are
        being sent at a steady rate. Only then is it safe to enable MSP OVERRIDE."""
        if not self.is_connected():
            return False
        return self._streaming.wait(timeout)

    def status(self):
        """Short summary for periodic status logging."""
        with self._lock:
            tracking = time.monotonic() - self._last_target_time < self._current_target_timeout()
            return (f"override {'ON' if self._overridden else 'off'}, "
                    f"tracking {'yes' if tracking else 'no'}, "
                    f"offsets p/y/t {self._offsets if tracking else (0, 0, 0)}, "
                    f"base throttle {self._base_throttle}, "
                    f"throttle integral {self._throttle_integral:.0f}")

    def disconnect(self):
        log.info("Disconnecting from FC")
        self._running = False
        self._streaming.clear()
        if self._thread:
            self._thread.join()
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()

    def run(self, error_x, error_y, area):
        """error_x: target offset from the image centre, -1 (left edge) .. 1 (right edge)
        error_y: target offset from the image centre, -1 (bottom edge) .. 1 (top edge)
        area: target box area / frame area"""
        # Target right -> yaw right, target above -> climb, target too small -> fly forward.
        # Roll stays centred - the horizontal error is corrected with yaw only.
        area_error = clamp((self.target_area - area) / self.target_area, -1.0, 1.0)
        yaw = self._clamp(self._apply_deadband(error_x) * self.yaw_gain, self.max_offset)
        pitch = self._clamp(self._apply_deadband(area_error) * self.pitch_gain, self.max_offset)
        vertical_error = self._apply_deadband(error_y)
        throttle = self._clamp(vertical_error * self.throttle_gain, self.max_throttle_offset)

        now = time.monotonic()
        with self._lock:
            interval = now - self._last_target_time
            if interval < self.max_target_timeout:
                if self._detection_interval is None:
                    self._detection_interval = interval
                else:
                    self._detection_interval += 0.2 * (interval - self._detection_interval)
                # Only integrate while the FC flies our commands, otherwise it winds up
                # before MSP OVERRIDE is switched on
                if self._overridden:
                    self._throttle_integral = clamp(
                        self._throttle_integral + vertical_error * self.throttle_i_gain * interval,
                        -self.max_throttle_integral, self.max_throttle_integral)
            self._offsets = (pitch, yaw, throttle)
            self._last_target_time = now

    def _apply_deadband(self, error):
        # Continuous at the deadband edge, so the command does not jump when leaving it
        if abs(error) <= self.deadband:
            return 0.0
        return (abs(error) - self.deadband) / (1.0 - self.deadband) * (1 if error > 0 else -1)

    @staticmethod
    def _clamp(value, limit):
        return int(clamp(value, -limit, limit))

    def _current_target_timeout(self):
        if self._detection_interval is None:
            return self.target_timeout
        return min(self.max_target_timeout, max(self.target_timeout, 2 * self._detection_interval))

    @staticmethod
    def _rc_is_valid(rc):
        return (rc is not None and len(rc) >= CHANNEL_COUNT
                and all(RC_VALID_MIN <= value <= RC_VALID_MAX for value in rc[:CHANNEL_COUNT]))

    def _build_frame(self, now):
        with self._lock:
            if now - self._last_target_time < self._current_target_timeout():
                pitch, yaw, throttle = self._offsets
            else:
                pitch = yaw = throttle = 0
            throttle += int(self._throttle_integral)

        # Throttle and AUX are mirrored from the FC, never hardcoded.
        # Throttle is the pilot's value minus a trim, because the stick position at the
        # moment of activation is usually above hover and the drone keeps climbing.
        data = list(self._last_rc)
        data[ROLL] = RC_MID
        data[PITCH] = RC_MID + pitch
        data[YAW] = RC_MID + yaw
        data[THROTTLE] = self._base_throttle - self.throttle_trim + throttle
        data = [clamp(int(value), RC_MIN, RC_MAX) for value in data]
        self._sent_throttles.append(data[THROTTLE])
        return data

    def _update_base_throttle(self, rc_throttle):
        # While override is active, MSP_RC returns the throttle we sent. Taking that as the
        # base would subtract the trim again on every frame and run the throttle down to
        # RC_MIN, so the base is only updated while MSP_RC reports something else (the RX).
        # Throttle changes while tracking and MSP_RC may lag a frame behind, so the reply is
        # compared with all recently sent values, not just the last one.
        overridden = rc_throttle in self._sent_throttles
        with self._lock:
            if overridden != self._overridden:
                if overridden:
                    log.info(f"MSP OVERRIDE active (pilot throttle {self._base_throttle})")
                else:
                    log.info(f"MSP OVERRIDE inactive - pilot in control "
                             f"(throttle integral was {self._throttle_integral:.0f})")
            self._overridden = overridden
            if not overridden:
                self._base_throttle = rc_throttle
                self._throttle_integral = 0.0   # pilot has control - start fresh next activation

    def _send_loop(self):
        try:
            self._send_loop_inner()
        except Exception:
            # The thread is a daemon - without this a crash would stop the RC stream silently
            log.exception("MSP send loop crashed - RC stream stopped")
            self._streaming.clear()
            raise

    def _send_loop_inner(self):
        sent_frames = 0
        invalid_rc_frames = 0
        next_send = time.monotonic()

        while self._running:
            # Send first, on a fixed schedule, so the rate does not depend on MSP_RC replies
            if self._last_rc is not None:
                self.send_control_signal(msp.MSP_SET_RAW_RC, self._build_frame(time.monotonic()))
                sent_frames += 1
                if sent_frames >= STREAM_READY_FRAMES and not self._streaming.is_set():
                    self._streaming.set()
                    log.info(f"MSP_SET_RAW_RC streaming at {1.0 / self.send_period:.0f} Hz "
                             f"- safe to enable MSP OVERRIDE")

            # Refresh mirrored channels, bounded by the time left until the next frame
            next_send += self.send_period
            invalid_rc_frames = self._poll_rc(max(0.0, next_send - time.monotonic()), invalid_rc_frames)

            delay = next_send - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            elif delay < -self.send_period:
                # Fell behind by more than a frame - resync instead of bursting
                next_send = time.monotonic()

    def _poll_rc(self, timeout, invalid_rc_frames):
        """Reads MSP_RC and updates the mirrored channels. Invalid readings (e.g. RX failsafe)
        are ignored and the last valid values are kept. Returns the updated count of
        consecutive invalid readings."""
        rc = self.get_rc_channels(timeout=timeout)
        rc_valid = self._rc_is_valid(rc)
        invalid_rc_frames = 0 if rc_valid else invalid_rc_frames + 1
        if rc_valid and self._rc_valid is not True:
            log.info(f"Valid RC data from FC: {rc[:CHANNEL_COUNT]}")
            self._rc_valid = True
        elif invalid_rc_frames >= RC_INVALID_LOG_FRAMES and self._rc_valid is not False:
            log.warning(f"No valid RC data from FC for {invalid_rc_frames} polls "
                        f"(failsafe / RX not bound?), last reading: {rc}")
            self._rc_valid = False
        if rc_valid:
            self._last_rc = rc[:CHANNEL_COUNT]
            self._update_base_throttle(rc[MSP_RC_THROTTLE])
        return invalid_rc_frames

    @staticmethod
    def get_checksum(msp_command_id, payload):
        checksum = 0
        for byte in bytes([len(payload), msp_command_id]) + payload:
            checksum ^= byte
        return checksum

    def send_control_signal(self, msp_command_id, data):
        payload = struct.pack(f'<{len(data)}H', *data)
        checksum = self.get_checksum(msp_command_id, payload)
        msp_package = MSP_REQUEST_HEADER + bytes([len(payload), msp_command_id]) + payload + bytes([checksum])
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.write(msp_package)

    def read_response(self, msp_command_id, timeout=0.1):
        # The port timeout is short, so the whole read never overruns `timeout`
        # (it bounds the MSP_RC poll that runs between MSP_SET_RAW_RC frames).
        deadline = time.monotonic() + timeout
        buffer = bytearray()
        while time.monotonic() < deadline:
            buffer += self.serial_port.read(self.serial_port.in_waiting or 1)

            while True:
                start = buffer.find(b'$M')
                if start < 0:
                    del buffer[:-1]     # keep a trailing '$' that may start a header
                    break
                del buffer[:start]
                if len(buffer) < 5:
                    break

                direction, size, cmd = buffer[2:5]
                if direction not in b'>!':
                    del buffer[:2]
                    continue
                if len(buffer) < 6 + size:
                    break

                payload = bytes(buffer[5:5 + size])
                checksum = buffer[5 + size]
                del buffer[:6 + size]

                # Skip errors, acks of other commands (e.g. MSP_SET_RAW_RC) and corrupted frames
                if direction == ord('!') or cmd != msp_command_id:
                    continue
                if checksum != self.get_checksum(cmd, payload):
                    continue
                return payload
        return None

    def get_rc_channels(self, timeout=0.1):
        self.serial_port.reset_input_buffer()
        self.send_control_signal(msp.MSP_RC, [])
        payload = self.read_response(msp.MSP_RC, timeout)
        if payload is None or len(payload) % 2:
            return None
        return list(struct.unpack(f'<{len(payload) // 2}H', payload))

    def get_FC_variant(self):
        self.send_control_signal(msp.MSP_BOARD_INFO, [])
        payload = self.read_response(msp.MSP_BOARD_INFO, timeout=0.5)
        if payload is None:
            return None
        return payload[:4].decode('ascii', errors='ignore')
