import os
import sys
import cv2
import glob

class Camera:
    def __init__(self, source_path, resolution=False, record=False):
        self.cam_idx = 0
        self.resize = False

        if 'usb' in source_path:
            self.source_type = 'usb'
            self.cam_idx = int(source_path[3:])
        elif 'picamera' in source_path:
            self.source_type = 'picamera'
            self.cam_idx = int(source_path[8:])
        elif 'webcam' in source_path:
            self.source_type = 'webcam'
            self.cam_idx = int(source_path[6:])
        else:
            print(f'Input {source_path} is invalid. Please try again.')
            sys.exit(0)

        if resolution:
            self.resize = True
            self.resW, self.resH = int(resolution.split('x')[0]), int(resolution.split('x')[1])

        if record:
            if self.source_type not in ['video', 'usb', 'webcam']:
                print('Recording only works for video and camera sources. Please try again.')
                sys.exit(0)
            if not resolution:
                print('Please specify resolution to record video at.')
                sys.exit(0)
            
            # Set up recording
            record_name = 'demo1.avi'
            record_fps = 30
            self.recorder = cv2.VideoWriter(record_name, cv2.VideoWriter_fourcc(*'MJPG'), record_fps, (self.resW, self.resH))

        if self.source_type == 'usb' or self.source_type == 'webcam':
                self.cap = cv2.VideoCapture(self.cam_idx)

                # Set camera or video resolution if specified by user
                if resolution:
                    ret = self.cap.set(3, self.resW)
                    ret = self.cap.set(4, self.resH)
        
        elif self.source_type == 'picamera':
            from picamera2 import Picamera2
            self.cap = Picamera2()
            self.cap.configure(self.cap.create_video_configuration(main={"format": 'XRGB8888', "size": (self.resW, self.resH)}))
            self.cap.start()

    def get_cap(self):
        if self.source_type == 'picamera':
            return self.cap.capture_array()
        return self.cap.read()