import cv2
import numpy as np
import tensorflow as tf
import time
from collections import deque
from datetime import datetime

MODEL_PATH = "waste_classifier.keras"

CAM_INDEX = 0
BACKEND = cv2.CAP_MSMF

IMG_SIZE = (224, 224)

CLASS_NAMES = ["huu_co", "tai_che", "vo_co"]

LABEL_ASCII = {
    "huu_co": "Huu co",
    "tai_che": "Tai che",
    "vo_co": "Vo co",
}

ROI_W_RATIO = 0.60
ROI_H_RATIO = 0.60

SMOOTH_WINDOW = 12
INFER_EVERY_N_FRAMES = 2

CONF_THRESH = 0.60


def load_model_compat(path: str):
    
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
    try:
        return tf.keras.models.load_model(
            path,
            compile=False,
            safe_mode=False,
            custom_objects={"preprocess_input": preprocess_input},
        )
    except TypeError:
        return tf.keras.models.load_model(
            path,
            compile=False,
            custom_objects={"preprocess_input": preprocess_input},
        )


model = load_model_compat(MODEL_PATH)


def pick_image_path():
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askopenfilename(
            title="Chon anh de phan loai",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All files", "*.*")],
        )
        root.destroy()
        return path
    except Exception as e:
        print("Upload image not available:", e)
        return ""


def preprocess_bgr(frame_bgr: np.ndarray) -> np.ndarray:
    """
    QUAN TRỌNG:
    - Model đã có Lambda(preprocess_input) bên trong rồi
    => ở đây CHỈ resize + BGR->RGB + float32 (0..255)
    - KHÔNG gọi mobilenet_v2.preprocess_input lần nữa (tránh preprocess 2 lần).
    """
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, IMG_SIZE, interpolation=cv2.INTER_AREA)
    x = resized.astype(np.float32)  
    return np.expand_dims(x, axis=0)


def predict_probs(frame_bgr: np.ndarray) -> np.ndarray:
    x = preprocess_bgr(frame_bgr)
    probs = model.predict(x, verbose=0)[0]
    return probs


def get_center_roi(frame: np.ndarray):
    h, w = frame.shape[:2]
    roi_w = int(w * ROI_W_RATIO)
    roi_h = int(h * ROI_H_RATIO)
    x1 = (w - roi_w) // 2
    y1 = (h - roi_h) // 2
    x2 = x1 + roi_w
    y2 = y1 + roi_h
    roi = frame[y1:y2, x1:x2]
    return (x1, y1, x2, y2), roi


def draw_text_bg(img: np.ndarray, text: str, org, scale=0.85, thickness=2,
                 text_color=(255, 255, 255), bg_color=(0, 0, 0), pad=6):
    x, y = org
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(img, (x - pad, y - th - pad), (x + tw + pad, y + baseline + pad), bg_color, -1)
    cv2.putText(img, text, (x, y), font, scale, text_color, thickness, cv2.LINE_AA)


def draw_clock_bottom_right(img: np.ndarray, margin=15):
    ts = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.75
    thickness = 2

    (tw, th), baseline = cv2.getTextSize(ts, font, scale, thickness)
    h, w = img.shape[:2]
    x = w - tw - margin
    y = h - margin

    draw_text_bg(img, ts, (x, y), scale=scale, thickness=thickness,
                 text_color=(255, 255, 255), bg_color=(0, 0, 0))


def format_result(probs: np.ndarray):
    idx = int(np.argmax(probs))
    key = CLASS_NAMES[idx]
    conf = float(probs[idx])
    name = LABEL_ASCII.get(key, key)
    if conf < CONF_THRESH:
        return f"Chua chac: {name} (conf={conf:.2f})", key, conf
    return f"{name} (conf={conf:.2f})", key, conf


cap = cv2.VideoCapture(CAM_INDEX, BACKEND)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam. Try another CAM_INDEX/backend.")

for _ in range(20):
    cap.read()
    time.sleep(0.03)

print("Controls:")
print("  Live is ON by default (real-time)")
print("  S = Scan (freeze current frame + show result)")
print("  R = Resume live")
print("  U = Upload image (chon 1 anh tu may)")
print("  Q = Quit")

frozen = False
frozen_raw = None
frozen_bbox = None
frozen_probs = None

prob_buffer = deque(maxlen=SMOOTH_WINDOW)
last_probs = None
infer_frame_counter = 0

frames_for_fps = 0
t_last = time.time()
fps = 0.0

while True:
    ret, raw = cap.read()
    if not ret or raw is None:
        break

    frames_for_fps += 1
    now = time.time()
    dt = now - t_last
    if dt >= 0.5:
        fps = frames_for_fps / dt
        frames_for_fps = 0
        t_last = now

    if not frozen:
        disp = raw.copy()
        (x1, y1, x2, y2), roi = get_center_roi(raw)

        cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 255), 2)

        infer_frame_counter += 1
        if (infer_frame_counter % INFER_EVERY_N_FRAMES) == 0 or last_probs is None:
            probs = predict_probs(roi)
            last_probs = probs
            prob_buffer.append(probs)

        avg_probs = np.mean(np.stack(prob_buffer, axis=0), axis=0) if len(prob_buffer) > 0 else last_probs
        text, _, _ = format_result(avg_probs)

        draw_text_bg(disp, f"REALTIME: {text}", (20, 40),
                     scale=0.85, text_color=(0, 255, 0), bg_color=(0, 0, 0))
        draw_text_bg(disp, f"FPS: {fps:.1f} | S=scan  U=upload  Q=quit", (20, 80),
                     scale=0.8, text_color=(255, 255, 255), bg_color=(0, 0, 0))

        draw_clock_bottom_right(disp)
        cv2.imshow("Waste Scanner", disp)

    else:
        show = frozen_raw.copy()
        (x1, y1, x2, y2) = frozen_bbox
        cv2.rectangle(show, (x1, y1), (x2, y2), (0, 255, 255), 2)

        text, _, _ = format_result(frozen_probs)
        draw_text_bg(show, f"SCANNED: {text}", (20, 40),
                     scale=0.9, text_color=(0, 255, 0), bg_color=(0, 0, 0))
        draw_text_bg(show, "Press R to resume", (20, show.shape[0] - 20),
                     scale=0.85, text_color=(0, 255, 255), bg_color=(0, 0, 0))

        top3 = np.argsort(frozen_probs)[::-1][:3]
        y = 90
        for j in top3:
            k = CLASS_NAMES[int(j)]
            name = LABEL_ASCII.get(k, k)
            draw_text_bg(show, f"{name}: {float(frozen_probs[int(j)]):.2f}", (20, y),
                         scale=0.8, text_color=(255, 255, 255), bg_color=(0, 0, 0))
            y += 40

        draw_clock_bottom_right(show)
        cv2.imshow("Waste Scanner", show)

    key = cv2.waitKey(1) & 0xFF

    if key in [ord("q"), ord("Q")]:
        break

    if key in [ord("u"), ord("U")]:
        path = pick_image_path()
        if path:
            img = cv2.imread(path)
            if img is None:
                print("Cannot read image:", path)
            else:
                show_img = img.copy()
                bbox, roi = get_center_roi(show_img)
                (x1, y1, x2, y2) = bbox
                cv2.rectangle(show_img, (x1, y1), (x2, y2), (0, 255, 255), 2)

                probs = predict_probs(roi)
                text, _, _ = format_result(probs)

                draw_text_bg(show_img, f"UPLOAD: {text}", (20, 40),
                             scale=0.85, text_color=(0, 255, 0), bg_color=(0, 0, 0))

                cv2.imshow("Uploaded Image - Waste Scanner", show_img)
                while True:
                    k2 = cv2.waitKey(0) & 0xFF
                    if k2 in [27, ord("q"), ord("Q"), ord("x"), ord("X")]:
                        break
                cv2.destroyWindow("Uploaded Image - Waste Scanner")

    if key in [ord("s"), ord("S")] and not frozen:
        frozen = True
        frozen_raw = raw.copy()
        bbox, roi = get_center_roi(frozen_raw)
        frozen_bbox = bbox
        frozen_probs = predict_probs(roi)

    if key in [ord("r"), ord("R")] and frozen:
        frozen = False
        frozen_raw = None
        frozen_bbox = None
        frozen_probs = None
        prob_buffer.clear()
        last_probs = None
        infer_frame_counter = 0

cap.release()
cv2.destroyAllWindows()