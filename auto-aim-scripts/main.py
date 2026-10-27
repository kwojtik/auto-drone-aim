import logging
import math
import os
import sys
import time
from collections import deque

import cv2
import numpy as np
from ultralytics import YOLO
# from yolox.tracker.byte_tracker import BYTETracker

from Camera.Camera import Camera
from Common.functions import parse_args, setup_logging
from Control.Control import Control


log = logging.getLogger('main')

# Bounding box colors (Tableau 10 color scheme)
BBOX_COLORS = [(164, 120, 87), (68, 148, 228), (93, 97, 209), (178, 182, 133), (88, 159, 106),
               (96, 202, 231), (159, 124, 168), (169, 162, 241), (98, 118, 150), (172, 176, 184)]
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
BLACK = (0, 0, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX

FPS_AVG_LEN = 200     # number of frames the average FPS is computed over
STATUS_PERIOD = 5.0   # seconds between status lines in the log


def draw_detection(frame, xmin, ymin, xmax, ymax, classname, classidx, conf):
    color = BBOX_COLORS[classidx % len(BBOX_COLORS)]
    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)

    label = f'{classname}: {int(conf * 100)}%'
    (label_w, label_h), baseline = cv2.getTextSize(label, FONT, 0.5, 1)
    label_ymin = max(ymin, label_h + 10)  # Don't draw the label too close to the top of the window
    cv2.rectangle(frame, (xmin, label_ymin - label_h - 10), (xmin + label_w, label_ymin + baseline - 10),
                  color, cv2.FILLED)
    cv2.putText(frame, label, (xmin, label_ymin - 7), FONT, 0.5, BLACK, 1)


def process_detections(frame, detections, labels, min_thresh):
    """Draw all detections above the threshold and return the largest one
    (the closest object) as (xmin, ymin, xmax, ymax, area), or None."""
    target = None
    for detection in detections:
        conf = detection.conf.item()
        if conf < min_thresh:
            continue

        # Ultralytics returns tensors - move to CPU and convert to ints
        xmin, ymin, xmax, ymax = detection.xyxy.cpu().numpy().squeeze().astype(int)
        classidx = int(detection.cls.item())
        draw_detection(frame, xmin, ymin, xmax, ymax, labels[classidx], classidx, conf)

        box_area = (xmax - xmin) * (ymax - ymin)
        if target is None or box_area > target[4]:
            target = (xmin, ymin, xmax, ymax, box_area)
    return target


def release_resources(camera, drone_control, record, headless):
    if drone_control is not None:
        drone_control.disconnect()
    if camera.source_type in ('webcam', 'usb'):
        camera.cap.release()
    elif camera.source_type == 'picamera':
        camera.cap.stop()
    if record:
        camera.recorder.release()
    if not headless:
        cv2.destroyAllWindows()


def main():
    args = parse_args()
    setup_logging(args.loglevel)

    if not os.path.exists(args.model):
        log.error('Model path is invalid or model was not found. Make sure the model filename was entered correctly.')
        sys.exit(0)

    log.info(f'Loading model {args.model}')
    model = YOLO(args.model, task='detect')
    labels = model.names

    camera = Camera(args.source, args.resolution, args.record)
    log.info(f'Camera source {args.source} ({camera.source_type}), resolution {camera.resW}x{camera.resH}')
    cam_centre = (camera.resW // 2, camera.resH // 2)
    cam_centre_x, cam_centre_y = cam_centre

    drone_control = None
    if not args.nocontrols:
        drone_control = Control(osd_detection=args.osd)
        # MSP OVERRIDE must only be enabled once the FC is already receiving
        # MSP_SET_RAW_RC at a steady rate with valid values on all channels.
        log.info('Waiting for MSP_SET_RAW_RC stream... (do NOT enable MSP OVERRIDE yet)')
        if not drone_control.wait_until_streaming(timeout=5.0):
            log.error('Could not start a steady MSP_SET_RAW_RC stream (no FC connection or no valid RC data). Exiting.')
            drone_control.disconnect()
            sys.exit(1)
        log.info('RC stream active - MSP OVERRIDE can now be enabled.')

    avg_frame_rate = 0
    frame_rate_buffer = deque(maxlen=FPS_AVG_LEN)
    last_status_time = time.monotonic()
    frames_with_target = 0
    frame_count = 0

    while True:
        t_start = time.perf_counter()

        try:
            frame = camera.get_frame()
        except Exception as e:
            log.exception(f'Error getting frame from source: {e}')
            break

        if camera.resize:
            frame = cv2.resize(frame, (camera.resW, camera.resH))

        results = model(frame, verbose=False)
        # results = model.track(frame, persist=True, tracker='bytetrack.yaml')

        cv2.circle(frame, cam_centre, 3, RED, -1)

        # Only one target drives the controls - the largest box (closest object)
        target = process_detections(frame, results[0].boxes, labels, args.thresh)

        if target is not None:
            xmin, ymin, xmax, ymax, box_area = target
            cx = (xmin + xmax) // 2
            cy = (ymin + ymax) // 2
            cv2.circle(frame, (cx, cy), 3, RED, -1)

            # Normalised errors: x is positive when the target is right of centre,
            # y is positive when it is above centre (image y grows downwards)
            error_x = (cx - cam_centre_x) / cam_centre_x
            error_y = (cam_centre_y - cy) / cam_centre_y
            area = box_area / (camera.resW * camera.resH)  # Simple distance estimate (smaller box = further away)
            if drone_control is not None:
                drone_control.run(error_x, error_y, area)

            distance_from_centre = math.dist((cx, cy), cam_centre)
            cv2.putText(frame, f'Distance from centre:{distance_from_centre:.0f}', (10, 20), FONT, .7, YELLOW, 2)

        if not args.headless:
            cv2.imshow('YOLO detection results', frame)
        if args.record:
            camera.recorder.write(frame)
        t_stop = time.perf_counter()

        if not args.headless:
            key = cv2.waitKey(5)
            if key in (ord('q'), ord('Q')):    # Quit
                break
            elif key in (ord('s'), ord('S')):  # Pause until any key is pressed
                cv2.waitKey()
            elif key in (ord('p'), ord('P')):  # Save a picture of this frame
                cv2.imwrite(f'../../images/frame_{int(time.time())}.png', frame)

        frame_rate_buffer.append(1 / (t_stop - t_start))
        avg_frame_rate = np.mean(frame_rate_buffer)

        # Periodic status line, so `systemctl status` shows the service is alive and tracking
        frame_count += 1
        if target is not None:
            frames_with_target += 1
        now = time.monotonic()
        if now - last_status_time >= STATUS_PERIOD:
            status = f'FPS {avg_frame_rate:.1f}, target in {frames_with_target}/{frame_count} frames'
            if drone_control is not None:
                status += f', {drone_control.status()}'
            log.info(status)
            last_status_time = now
            frames_with_target = 0
            frame_count = 0

    log.info(f'Average pipeline FPS: {avg_frame_rate:.2f}')
    release_resources(camera, drone_control, args.record, args.headless)


if __name__ == '__main__':
    main()
