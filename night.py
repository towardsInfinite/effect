
import cv2
import numpy as np
import random

cap = cv2.VideoCapture(0)

frame_count = 0
hologram_on = True

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    frame_count += 1

    if hologram_on:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        hologram = np.zeros_like(frame)
        hologram[:, :, 0] = gray
        hologram[:, :, 1] = gray // 2

        edges = cv2.Canny(gray, 60, 120)
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        edges[:, :, 0] = edges[:, :, 0]
        edges[:, :, 1] = edges[:, :, 1]
        edges[:, :, 2] = 0

        glow = cv2.GaussianBlur(edges, (15, 15), 0)
        hologram = cv2.addWeighted(
            hologram, 0.8, glow, 0.7, 0
        )

        for y in range(frame_count % 6, h, 6):
            cv2.line(
                hologram, (0, y), (w, y),
                (255, 100, 0), 1
            )

        # 6. Random hologram flicker
        if random.randint(1, 15) == 1:
            hologram = (
                hologram * 0.5
            ).astype(np.uint8)

        # 7. Add hologram label
        cv2.putText(
            hologram, "HOLOGRAM ACTIVE",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (255, 255, 0), 2
        )

        frame = hologram

    # Display instructions
    cv2.putText(
        frame, "H: Toggle | Q: Quit",
        (20, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6, (255, 255, 255), 2
    )

    cv2.imshow("HOLOGRAM CAMERA", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('h'):
        hologram_on = not hologram_on

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
