import serial
import struct
import msp_helper as msp


class Control:
    def __init__(self, companion_computer="COM3", baud_rate=115200):
        self.default_roll = 1500
        self.default_pitch = 1500
        self.default_yaw = 1500
        self.default_throttle = 1000
        self.default_servo_aux2 = 1000

        try:
            self.serial_port = serial.Serial(companion_computer, baud_rate, timeout=1)
        except serial.SerialException as e:
            print(f"Error connecting to serial port: {e}")

    def disconnect(self):
        self.serial_port.close()

    def run(self, distanceX, distanceY, distanceZ):
        roll = self.default_roll + int(distanceY * 0.5)
        pitch = int(self.default_pitch * distanceZ)
        yaw = self.default_yaw + int(distanceX * 0.5)
        throttle = self.default_throttle
        servo_aux2 = self.default_servo_aux2

        data = [roll, pitch, yaw, 0, throttle, servo_aux2, 0, 0]
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
        self.serial_port.write(msp_package)
