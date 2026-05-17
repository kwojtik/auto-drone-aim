import serial


class Control:
    def __init__(self, companion_computer, baud_rate=115200):
        self.serial_port = serial.Serial(companion_computer, baud_rate, timeout=1)
    
    def disconnect(self):
        self.serial_port.close()

    def run(self, distanceX, distanceY):
        self.send_control_signal((distanceX, distanceY))

    def send_control_signal(self, signal):
        pass
