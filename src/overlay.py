import cv2
import numpy as np


def overlay_image(background, overlay, x, y, w, h):
    if overlay is None:
        return background

    overlay = cv2.resize(overlay, (w, h))

    if overlay.shape[2] == 4:
        b, g, r, a = cv2.split(overlay)
        overlay_rgb = cv2.merge((b, g, r))
        mask = (a / 255.0)[:, :, None]    
    else:
        overlay_rgb = overlay
        mask = 1.0                          

    h_img, w_img = overlay_rgb.shape[:2]

    y = max(0, min(y, background.shape[0] - h_img))
    x = max(0, min(x, background.shape[1] - w_img))

    roi = background[y:y+h_img, x:x+w_img]

    blended = (mask * overlay_rgb + (1 - mask) * roi).astype(np.uint8)
    background[y:y+h_img, x:x+w_img] = blended

    return background