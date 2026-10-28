import argparse
import logging
import os
import sys


# sd-daemon(3) priority prefixes - journald strips them and stores the record at that priority
SYSLOG_PRIORITIES = {
    logging.CRITICAL: 2,
    logging.ERROR: 3,
    logging.WARNING: 4,
    logging.INFO: 6,
    logging.DEBUG: 7,
}


class JournalFormatter(logging.Formatter):
    def format(self, record):
        priority = SYSLOG_PRIORITIES.get(record.levelno, 6)
        return f'<{priority}>{super().format(record)}'


def setup_logging(level='INFO'):
    handler = logging.StreamHandler(sys.stderr)
    # Under systemd the journal adds the timestamp and process name itself
    if os.environ.get('JOURNAL_STREAM'):
        handler.setFormatter(JournalFormatter('%(name)s: %(message)s'))
    else:
        handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s'))
    logging.basicConfig(level=level.upper(), handlers=[handler], force=True)



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='Path to YOLO model file (example: "runs/detect/train/weights/best.pt")',
                        required=True)
    parser.add_argument('--source', help='Image source, can be image file ("test.jpg"), \
                        image folder ("test_dir"), video file ("testvid.mp4"), or index of USB camera ("usb0")', 
                        required=True)
    parser.add_argument('--thresh', help='Minimum confidence threshold for detected objects to be drawn and tracked (example: "0.4")',
                        type=float, default=0.8)
    parser.add_argument('--resolution', help='Resolution in WxH to display inference results at (example: "640x480"), \
                        otherwise, match source resolution',
                        default=None)
    parser.add_argument('--record', help='Record results from video or webcam and save it as "demo1.avi". Must specify --resolution argument to record.',
                        action='store_true')
    parser.add_argument('--headless', help='Skip the live preview window (required when running without a display, e.g. as a systemd service).',
                        action='store_true')
    parser.add_argument('--nocontrols', help='Disable sending RC commands to the flight controller (for testing purposes).',
                        action='store_true')
    parser.add_argument('--loglevel', help='Logging level: DEBUG, INFO, WARNING or ERROR (logs go to stderr / the systemd journal).',
                        default='INFO')

    return parser.parse_args()
