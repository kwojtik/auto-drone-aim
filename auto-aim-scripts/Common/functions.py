import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='Path to YOLO model file (example: "runs/detect/train/weights/best.pt")',
                        required=True)
    parser.add_argument('--source', help='Image source, can be image file ("test.jpg"), \
                        image folder ("test_dir"), video file ("testvid.mp4"), or index of USB camera ("usb0")', 
                        required=True)
    parser.add_argument('--thresh', help='Minimum confidence threshold for displaying detected objects (example: "0.4")',
                        default=0.5)
    parser.add_argument('--resolution', help='Resolution in WxH to display inference results at (example: "640x480"), \
                        otherwise, match source resolution',
                        default=None)
    parser.add_argument('--record', help='Record results from video or webcam and save it as "demo1.avi". Must specify --resolution argument to record.',
                        action='store_true')

    return parser.parse_args()
