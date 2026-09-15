import cv2
import os
import time
from datetime import datetime
from detector import detect_human_face, crop
from expressions import get_blendshapes, Calibrator, classify, POSE_TO_CAT
from overlay import overlay_image

os.makedirs("captures", exist_ok=True)

moods_images = {}
calibrator = Calibrator(n_frames=60)
calibrating = False
means = None
stds = None

# reaction state
REACTION_DELAY = 0.4
OVERLAY_DURATION = 0.8

last_pose = None
pose_since = 0.0
overlay_until = 0.0
reacted_to_current_pose = False

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

while True:
    ok, frame = cap.read()
    if not ok:
        break

    # calibration mode 
    if calibrating:
        bs = get_blendshapes(frame)
        if bs:
            calibrator.add(bs)
            cv2.putText(
                frame,
                f"Calibrating... {len(calibrator.samples)}/{calibrator.n_frames}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 225, 255), 2,
            )
            if calibrator.is_done():
                means, stds = calibrator.compute()
                calibrating = False
                print("Calibration done.")

        cv2.imshow("cat mood detector", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        continue

    # normal mode 
    frame, box = detect_human_face(frame)

    if box and means is not None:
        bs = get_blendshapes(frame)
        if bs:
            pose = classify(bs, means, stds)
            now = time.time()

            if pose != "neutral":
                if pose != last_pose:
                    pose_since = now
                    last_pose = pose
                    reacted_to_current_pose = False

                elapsed = now - pose_since

                if elapsed >= REACTION_DELAY and not reacted_to_current_pose:
                    overlay_until = now + OVERLAY_DURATION
                    reacted_to_current_pose = True
            else:
                last_pose = None
                pose_since = 0.0
                reacted_to_current_pose = False

            # draw overlay while its timer is active
            if now < overlay_until:
                mood_file = POSE_TO_CAT.get(pose, "moods/smug.jpg")
                if mood_file not in moods_images:
                    moods_images[mood_file] = cv2.imread(mood_file, cv2.IMREAD_UNCHANGED)

                img = moods_images[mood_file]
                if img is not None:
                    box_w = box[2] - box[0]
                    box_h = box[3] - box[1]
                    overlay_image(frame, img, box[0], box[1], box_w, box_h)

            cv2.putText(frame, pose, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 225, 255), 2)

    cv2.imshow("cat mood detector", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    elif key == ord('c'):
        calibrator.reset()
        calibrating = True
        means = None
        stds = None
        overlay_until = 0.0
        last_pose = None
        pose_since = 0.0
        reacted_to_current_pose = False
        print("Calibrating, look bored or don't pose at all for 2 seconds...")

    elif key == ord('s'):
        ts = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        path = f"captures/mood_{ts}.png"
        cv2.imwrite(path, frame)
        print(f"saved {path}")

cap.release()
cv2.destroyAllWindows()