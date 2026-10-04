import serial
import struct
import Control.msp_helper as msp


class Control:
    def __init__(self, companion_computer="/dev/ttyACM0", baud_rate=115200):
        self.default_roll = 1500
        self.default_pitch = 1500
        self.default_yaw = 1500
        self.default_throttle = 1000
        self.serial_port = None

        try:
            self.serial_port = serial.Serial(companion_computer, baud_rate, timeout=1)
        except serial.SerialException as e:
            print(f"Error connecting to serial port: {e}")
            return

        print(f"FC Variant: {self.get_FC_variant()}")

    def disconnect(self):
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()

    def run(self, distanceX, distanceY, distanceZ):
        roll = self.default_roll + int(distanceY * 0.5)
        pitch = int(self.default_pitch * distanceZ)
        yaw = self.default_yaw + int(distanceX * 0.5)
        throttle = self.default_throttle

        data = [roll, pitch, yaw, 0, throttle, 0, 0, 0]
        print(data)
        self.send_control_signal(msp.MSP_SET_RAW_RC, data)

    def get_checksum(self, msp_command_id, payload):
        checksum = 0
        length = len(payload)
        for byte in bytes([length, msp_command_id]) + payload:
            checksum ^= byte

        checksum &= 0xFF
        return checksum

    def send_control_signal(self, msp_command_id, data):
        payload = bytearray()
        for value in data:
            payload += struct.pack('<H', value)  
        
        header = b'$M<'
        length = len(payload)
        checksum = self.get_checksum(msp_command_id, payload)
        
        msp_package = header + bytes([length, msp_command_id]) + payload + bytes([checksum])
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.write(msp_package)

    def get_FC_variant(self):
        self.send_control_signal(msp.MSP_BOARD_INFO, [])
        response = self.serial_port.read(20).decode('utf-8', errors='ignore')
        return response
