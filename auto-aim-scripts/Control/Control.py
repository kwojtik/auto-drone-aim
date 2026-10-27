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
# Permanent box ID of the MSP OVERRIDE mode (same in Betaflight and INAV). Its position in
# the MSP_STATUS mode bitmask is its index in the MSP_BOXIDS list.
BOX_MSP_OVERRIDE = 50
# MSP_STATUS is polled every this many frames (10 Hz at 50 Hz), after MSP_RC
STATUS_POLL_FRAMES = 5
# Consecutive MSP_RC readings without our throttle while the mode is on before it is logged
ECHO_MISS_LOG_FRAMES = 25
# Detection state is shown on the OSD through the craft name (OSD element "Craft name").
# The FC keeps at most 16 characters, and analog OSD fonts only have uppercase letters.
OSD_TEXT_PREFIX = 'DETECTED: '
OSD_NAME_MAX_LENGTH = 16
# The craft name is re-sent this often even without a change, in case a write was lost
# or the FC rebooted (seconds)
OSD_REFRESH_PERIOD = 1.0


def clamp(value, low, high):
    return max(low, min(high, value))


class Control:
    def __init__(self, companion_computer="/dev/ttyACM0", baud_rate=115200,
                 send_rate_hz=50, target_timeout=0.3, max_target_timeout=1.0,
                 max_offset=200, max_throttle_offset=25, throttle_trim=50,
                 throttle_i_gain=50, max_throttle_integral=150,
                 target_area=0.1, deadband=0.05, osd_detection=False):
        # Gains are stick deflection (us) for a full normalised error (1.0)
        self.yaw_gain = 200
        self.pitch_gain = 150
        self.throttle_gain = 100
        self.max_offset = max_offset            # max pitch / yaw deflection from centre
        self.max_throttle_offset = max_throttle_offset
        self.target_area = target_area          # box area / frame area to hold the target at
        self.deadband = deadband                # normalised error ignored around zero
        self.send_period = 1.0 / max(send_rate_hz, MIN_SEND_RATE_HZ)
        self.osd_detection = osd_detection      # show the detection state in the OSD craft name
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
        self._overridden = False                # MSP OVERRIDE mode active on the FC
        self._override_box_index = None         # bit of MSP OVERRIDE in the MSP_STATUS mode flags
        self._override_mode = None              # last mode flag read from MSP_STATUS (None = unknown)
        self._echo_misses = 0
        self._fc_variant = None
        self._last_target_time = 0.0
        self._detection_interval = None         # smoothed time between detections
        self._last_rc = None
        self._base_throttle = None              # pilot's throttle from the RX (before override)
        self._sent_throttles = deque(maxlen=SENT_THROTTLE_HISTORY)
        self._running = False
        self._thread = None
        self._streaming = threading.Event()     # set once MSP_SET_RAW_RC is streamed steadily
        self._rc_valid = None                   # last MSP_RC validity, to log only changes
        self._original_name = None              # craft name before the OSD text, restored on exit
        self._osd_text = None                   # last text written to the craft name
        self._osd_sent_time = 0.0

        try:
            self.serial_port = serial.Serial(companion_computer, baud_rate, timeout=0.002)
        except serial.SerialException as e:
            log.error(f"Error connecting to serial port {companion_computer}: {e}")
            return

        self._fc_variant = self.get_FC_variant()
        log.info(f"Connected to {companion_computer} @ {baud_rate}, FC variant: {self._fc_variant}")

        if self.osd_detection:
            # A name starting with our prefix is left over from a run that did not exit cleanly
            self._original_name = self.get_craft_name()
            if self._original_name is not None and self._original_name.startswith(OSD_TEXT_PREFIX):
                self._original_name = ''
            log.info(f"Craft name: '{self._original_name}' - replaced by the detection state on the OSD")

        # Override is detected from the FC's mode flags. Only if the box cannot be found does
        # it fall back to recognising our throttle echoed back in MSP_RC.
        self._override_box_index = self.get_box_index(BOX_MSP_OVERRIDE)
        if self._override_box_index is None:
            log.warning("MSP OVERRIDE mode not found in MSP_BOXIDS (firmware without "
                        "USE_MSP_OVERRIDE, msp_override_channels_mask = 0, or no reply) - "
                        "falling back to detecting override from the MSP_RC throttle echo")
        elif self._override_box_index >= 32 and self._fc_variant != 'BTFL':
            log.warning(f"MSP OVERRIDE is mode bit {self._override_box_index}, beyond the 32 bits "
                        f"MSP_STATUS reports on {self._fc_variant} - falling back to MSP_RC echo")
            self._override_box_index = None
        else:
            log.info(f"MSP OVERRIDE is mode bit {self._override_box_index}")

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
            if self._original_name is not None:
                self.set_craft_name(self._original_name)
            self.serial_port.close()

    def is_target_detected(self):
        """True while a target was seen within the target timeout - the same hold that keeps
        the commands active, so the OSD does not flicker between single missed detections."""
        with self._lock:
            return time.monotonic() - self._last_target_time < self._current_target_timeout()

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
            log.info(f"Offsest p/y/t: {pitch}/{yaw}/{throttle}");
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
        echoed = rc_throttle in self._sent_throttles
        with self._lock:
            overridden = self._override_mode if self._override_mode is not None else echoed
            if overridden != self._overridden:
                if overridden:
                    log.info(f"MSP OVERRIDE active (pilot throttle {self._base_throttle})")
                else:
                    log.info(f"MSP OVERRIDE inactive - pilot in control "
                             f"(throttle integral was {self._throttle_integral:.0f})")
                self._echo_misses = 0
            self._overridden = overridden

            if overridden and not echoed:
                self._echo_misses += 1
                if self._echo_misses == ECHO_MISS_LOG_FRAMES:
                    log.warning(f"MSP OVERRIDE is on but MSP_RC throttle ({rc_throttle}) is not "
                                f"what we send - is the throttle channel in msp_override_channels_mask?")
            else:
                self._echo_misses = 0

            if not overridden:
                # MSP_STATUS is polled less often than MSP_RC, so right after the switch is
                # flipped MSP_RC can already echo our value - never take that as the base
                if not echoed:
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
        loop_count = 0
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
            loop_count += 1
            if self._override_box_index is not None and loop_count % STATUS_POLL_FRAMES == 0:
                self._poll_override_mode(max(0.0, next_send - time.monotonic()))
            if self.osd_detection:
                self._update_osd()

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

    def _update_osd(self):
        """Writes the detection state to the craft name when it changes, and periodically.
        Runs in the send loop, so it never interleaves with the MSP_SET_RAW_RC frames."""
        text = OSD_TEXT_PREFIX + ('TRUE' if self.is_target_detected() else 'FALSE')
        now = time.monotonic()
        if text != self._osd_text or now - self._osd_sent_time >= OSD_REFRESH_PERIOD:
            if text != self._osd_text:
                log.info(f"OSD: {text}")
            self.set_craft_name(text)
            self._osd_text = text
            self._osd_sent_time = now

    def _poll_override_mode(self, timeout):
        """Reads the MSP OVERRIDE mode flag. A missed reply keeps the last known state."""
        active = self.get_mode_active(self._override_box_index, timeout=timeout)
        if active is not None:
            with self._lock:
                self._override_mode = active

    @staticmethod
    def get_checksum(msp_command_id, payload):
        checksum = 0
        for byte in bytes([len(payload), msp_command_id]) + payload:
            checksum ^= byte
        return checksum

    def send_control_signal(self, msp_command_id, data):
        self.send_raw(msp_command_id, struct.pack(f'<{len(data)}H', *data))

    def send_raw(self, msp_command_id, payload):
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

    def get_box_index(self, permanent_id, timeout=0.5):
        """Position of a mode in the FC's active box list, which is also its bit in the
        MSP_STATUS mode flags. None if the mode is not available or there was no reply."""
        self.send_control_signal(msp.MSP_BOXIDS, [])
        payload = self.read_response(msp.MSP_BOXIDS, timeout)
        if payload is None or permanent_id not in payload:
            return None
        return payload.index(permanent_id)

    def get_mode_active(self, box_index, timeout=0.1):
        """True / False from the MSP_STATUS mode flags, None if there was no valid reply."""
        self.serial_port.reset_input_buffer()
        self.send_control_signal(msp.MSP_STATUS, [])
        payload = self.read_response(msp.MSP_STATUS, timeout)
        if payload is None or len(payload) < 10:
            return None
        # u16 cycle time, u16 i2c errors, u16 sensors, u32 first 32 mode bits, ...
        flags = int.from_bytes(payload[6:10], 'little')
        # Betaflight appends the remaining mode bits: u8 byte count at offset 15, then the bytes
        if box_index >= 32:
            if len(payload) < 16:
                return None
            extra = payload[16:16 + payload[15]]
            flags |= int.from_bytes(extra, 'little') << 32
        return bool(flags >> box_index & 1)

    def get_craft_name(self, timeout=0.5):
        """Craft name from the FC ('' if not set), None if there was no reply."""
        self.serial_port.reset_input_buffer()
        self.send_control_signal(msp.MSP_NAME, [])
        payload = self.read_response(msp.MSP_NAME, timeout)
        if payload is None:
            return None
        return payload.decode('ascii', errors='ignore').rstrip('\0')

    def set_craft_name(self, name):
        # Only changes the name in RAM - it is not saved unless MSP_EEPROM_WRITE is sent
        self.send_raw(msp.MSP_SET_NAME, name.upper().encode('ascii', errors='ignore')[:OSD_NAME_MAX_LENGTH])

    def get_FC_variant(self):
        self.send_control_signal(msp.MSP_BOARD_INFO, [])
        payload = self.read_response(msp.MSP_BOARD_INFO, timeout=0.5)
        if payload is None:
            return None
        return payload[:4].decode('ascii', errors='ignore')
