"""
Nine-Hole Peg Test (9HPT) – Robotski vid izziv
Določanje kinematičnih parametrov gibanja roke iz enega pogleda.

Uporaba:
    python src/main.py --input data/video.mp4 --output data/results/
    python src/main.py --input 0   (webcam)
"""

import cv2
import mediapipe as mp
import numpy as np
import argparse
import os
import json
import csv
from collections import deque
import time

# ── Konstante ─────────────────────────────────────────────────────────────────

WRIST       = 0
THUMB_TIP   = 4
INDEX_TIP   = 8
MIDDLE_TIP  = 12
RING_TIP    = 16
PINKY_TIP   = 20

SMOOTH_WINDOW = 5

COLOR_HAND   = (0, 255, 0)
COLOR_THUMB  = (255, 100,   0)
COLOR_INDEX  = (0,   100, 255)
COLOR_TRAIL  = (200, 200,   0)
COLOR_TEXT   = (255, 255, 255)
COLOR_GRIP   = (0, 0, 255)
COLOR_RELEASE= (0, 255, 0)

# 9HPT plošča – znane dimenzije
HOLE_DIAMETER_MM   = 10.0
HOLE_GRID_ROWS     = 3
HOLE_GRID_COLS     = 3
BOARD_WIDTH_MM     = 320.0
BOARD_HEIGHT_MM    = 130.0

# Prag za zaznavanje prijema (razdalja med palcem in kazalcem v mm)
GRIP_THRESHOLD_MM  = 35.0

# Koliko okvirjev na začetku uporabimo za umerjanje
CALIB_FRAMES       = 50

# ── Pomožne funkcije ───────────────────────────────────────────────────────────

def pixel_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def smooth(values, window=SMOOTH_WINDOW):
    if len(values) < window:
        return values[-1] if values else 0.0
    return float(np.mean(list(values)[-window:]))


def draw_trail(frame, trail, color, max_len=60):
    pts = list(trail)[-max_len:]
    for i in range(1, len(pts)):
        if pts[i-1] is not None and pts[i] is not None:
            alpha = i / len(pts)
            c = tuple(int(v * alpha) for v in color)
            cv2.line(frame, pts[i-1], pts[i], c, 2)


def overlay_text(frame, lines, x=10, y=20, dy=22):
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (x, y + i * dy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (x, y + i * dy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_TEXT, 1, cv2.LINE_AA)


# ── Zaznavanje lukenj za umerjanje ────────────────────────────────────────────

def detect_holes(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    params = cv2.SimpleBlobDetector_Params()
    params.filterByColor       = True
    params.blobColor           = 255
    params.filterByArea        = True
    params.minArea             = 40        # bilo 50 - znižamo
    params.maxArea             = 600       # bilo 3000 - znižamo, luknje so ~50-70
    params.filterByCircularity = True
    params.minCircularity      = 0.4       # bilo 0.6 - znižamo ker so luknje 9x10
    params.filterByConvexity   = True
    params.minConvexity        = 0.6       # bilo 0.7
    params.filterByInertia     = True
    params.minInertiaRatio     = 0.3       # bilo 0.4

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(gray)

    centers = [(int(kp.pt[0]), int(kp.pt[1])) for kp in keypoints]
    return centers


def estimate_px_per_mm(frames, known_spacing_mm=None):
    """
    Iz prvih N okvirjev oceni px_per_mm.
    Zazna luknje, vzame mediano razdalj med sosednjimi luknjami.
    known_spacing_mm: razdalja med luknjami v mm (izmerimo iz CAD/opisa)
    """
    all_centers = []
    for frame in frames:
        centers = detect_holes(frame)
        all_centers.extend(centers)

    if len(all_centers) < 4:
        return None, []

    # Povpreči pozicije z mediano (robustno na outlierje)
    pts = np.array(all_centers, dtype=np.float32)

    # Razdalje med vsemi pari točk
    dists = []
    for i in range(len(pts)):
        for j in range(i+1, len(pts)):
            d = np.linalg.norm(pts[i] - pts[j])
            dists.append(d)

    if not dists:
        return None, all_centers

    dists = np.array(dists)

    # Najdi skupino razdalj ki ustreza sosednjim luknjam
    # (predpostavljamo da je razdalja med sosednjimi luknjami
    #  manjša kot diagonalna razdalja)
    min_d = np.min(dists)
    # Sosednje razdalje so vse v rangu min_d * 1.5
    neighbor_dists = dists[dists < min_d * 1.6]

    if len(neighbor_dists) == 0:
        return None, all_centers

    spacing_px = float(np.median(neighbor_dists))

    if known_spacing_mm is None:
        # Ocenimo spacing iz dimenzij plošče:
        # Plošča 320x130mm, 3x3 luknje
        # Spacing ~ (130mm - 2*okvirja) / 2 ≈ 32mm (tipično za 9HPT)
        known_spacing_mm = 32.0

    px_per_mm = spacing_px / known_spacing_mm
    return px_per_mm, all_centers


# ── Zaznavanje prijema/spusta pina ────────────────────────────────────────────

class GripDetector:
    """Zazna prijem in spust pina iz razdalje med palcem in kazalcem."""

    def __init__(self, threshold_mm=GRIP_THRESHOLD_MM, px_per_mm=1.0):
        self.threshold_px = threshold_mm * px_per_mm
        self.is_gripping  = False
        self.grip_events  = []   # seznam (frame, time, 'grip'/'release')
        self._prev_dist   = None

    def update(self, thumb_px, index_px, frame_idx, time_s):
        dist = pixel_distance(thumb_px, index_px)

        gripping_now = dist < self.threshold_px

        if gripping_now != self.is_gripping:
            event_type = "grip" if gripping_now else "release"
            self.grip_events.append({
                "frame": frame_idx,
                "time_s": round(time_s, 3),
                "type": event_type,
                "distance_px": round(dist, 1)
            })
            self.is_gripping = gripping_now

        self._prev_dist = dist
        return dist


# ── Razred za sledenje kinematike ─────────────────────────────────────────────

class KinematicsTracker:
    def __init__(self, name, fps):
        self.name = name
        self.fps  = fps
        self.dt   = 1.0 / fps

        self.positions    = []
        self.path_length  = []
        self.velocity     = []
        self.accel        = []
        self.timestamps   = []

        self._vel_buf  = deque(maxlen=SMOOTH_WINDOW)
        self._acc_buf  = deque(maxlen=SMOOTH_WINDOW)
        self._trail    = deque(maxlen=120)
        self._cum_dist = 0.0
        self._frame    = 0

    def update(self, pos):
        t = self._frame * self.dt
        self.timestamps.append(t)
        self.positions.append(pos)
        self._trail.append(pos)
        self._frame += 1

        if len(self.positions) >= 2:
            d = pixel_distance(self.positions[-1], self.positions[-2])
            self._cum_dist += d
            v = d / self.dt
            self._vel_buf.append(v)
            v_smooth = smooth(self._vel_buf)
            self.velocity.append(v_smooth)

            if len(self.velocity) >= 2:
                a = (self.velocity[-1] - self.velocity[-2]) / self.dt
                self._acc_buf.append(a)
                self.accel.append(smooth(self._acc_buf))
            else:
                self.accel.append(0.0)
        else:
            self.velocity.append(0.0)
            self.accel.append(0.0)

        self.path_length.append(self._cum_dist)

    def current_stats(self):
        v = self.velocity[-1]    if self.velocity    else 0.0
        a = self.accel[-1]       if self.accel       else 0.0
        d = self.path_length[-1] if self.path_length else 0.0
        return d, v, a

    @property
    def trail(self):
        return self._trail

    def to_dict(self):
        return {
            "name":         self.name,
            "fps":          self.fps,
            "timestamps":   self.timestamps,
            "positions":    self.positions,
            "path_length":  self.path_length,
            "velocity":     self.velocity,
            "acceleration": self.accel,
        }


# ── Glavna funkcija ───────────────────────────────────────────────────────────

def process_video(input_source, output_dir, px_per_mm=None, show=True):
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(input_source)
    if not cap.isOpened():
        raise RuntimeError(f"Ne morem odpreti vira: {input_source}")

    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] Vir: {input_source}  |  {W}×{H} @ {fps:.1f} fps  |  {total} okvirjev")

    # ── Faza 1: umerjanje iz prvih CALIB_FRAMES okvirjev ─────────────────────
    calib_frames = []
    while len(calib_frames) < CALIB_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
        calib_frames.append(frame.copy())

    detected_px_per_mm = None
    hole_centers       = []

    if px_per_mm is None:
        print(f"[INFO] Umerjanje iz prvih {len(calib_frames)} okvirjev...")
        detected_px_per_mm, hole_centers = estimate_px_per_mm(calib_frames)
        if detected_px_per_mm:
            px_per_mm = detected_px_per_mm
            print(f"[INFO] Zaznane luknje: {len(hole_centers)}  |  px_per_mm = {px_per_mm:.3f}")
        else:
            px_per_mm = 1.0
            print(f"[WARN] Umerjanje ni uspelo – luknje niso zaznane. Uporabljam px_per_mm=1.0")
    else:
        print(f"[INFO] Ročni px_per_mm = {px_per_mm:.3f}")

    # ── Video writer ──────────────────────────────────────────────────────────
    out_path = os.path.join(output_dir, "annotated.mp4")
    fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
    writer   = cv2.VideoWriter(out_path, fourcc, fps, (W, H))

    # ── MediaPipe ─────────────────────────────────────────────────────────────
    mp_hands  = mp.solutions.hands
    mp_draw   = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )

    # ── Trackerji in grip detektor ────────────────────────────────────────────
    wrist_t = KinematicsTracker("wrist", fps)
    thumb_t = KinematicsTracker("thumb", fps)
    index_t = KinematicsTracker("index", fps)
    grip    = GripDetector(threshold_mm=GRIP_THRESHOLD_MM, px_per_mm=px_per_mm)

    # Najprej obdelaj kalibracijske okvirje
    all_frames = calib_frames
    frame_idx  = 0
    t0         = time.time()

    def process_frame(frame, frame_idx):
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        # Nariši zaznane luknje na prvih okvirjih
        if frame_idx < CALIB_FRAMES and hole_centers:
            for cx, cy in hole_centers:
                cv2.circle(frame, (cx, cy), 8, (0, 255, 255), 2)

        detected = False
        if result.multi_hand_landmarks:
            for hand_lm in result.multi_hand_landmarks:
                detected = True

                mp_draw.draw_landmarks(
                    frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style()
                )

                def lm_px(idx):
                    lm = hand_lm.landmark[idx]
                    return (int(lm.x * W), int(lm.y * H))

                wrist_px = lm_px(WRIST)
                thumb_px = lm_px(THUMB_TIP)
                index_px = lm_px(INDEX_TIP)

                wrist_t.update(wrist_px)
                thumb_t.update(thumb_px)
                index_t.update(index_px)

                t_s      = frame_idx / fps
                dist_px  = grip.update(thumb_px, index_px, frame_idx, t_s)
                dist_mm  = dist_px / px_per_mm

                draw_trail(frame, wrist_t.trail, COLOR_TRAIL)
                draw_trail(frame, thumb_t.trail, COLOR_THUMB)
                draw_trail(frame, index_t.trail, COLOR_INDEX)

                cv2.circle(frame, thumb_px, 8, COLOR_THUMB, -1)
                cv2.circle(frame, index_px, 8, COLOR_INDEX, -1)
                cv2.circle(frame, wrist_px, 6, COLOR_HAND,  -1)

                # Linija med palcem in kazalcem + barva glede na prijem
                grip_color = COLOR_GRIP if grip.is_gripping else COLOR_RELEASE
                cv2.line(frame, thumb_px, index_px, grip_color, 2)

                # Indikator prijema
                grip_text  = "PRIJEM" if grip.is_gripping else "SPUST"
                grip_label_color = (0, 0, 255) if grip.is_gripping else (0, 200, 0)
                cv2.putText(frame, grip_text,
                            (W - 160, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                            (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(frame, grip_text,
                            (W - 160, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                            grip_label_color, 2, cv2.LINE_AA)

                wd, wv, wa   = wrist_t.current_stats()
                td, tv, ta   = thumb_t.current_stats()
                id_, iv, ia  = index_t.current_stats()

                lines = [
                    f"Frame: {frame_idx:5d}   t={frame_idx/fps:.2f}s",
                    f"px/mm={px_per_mm:.2f}  d(thumb-idx)={dist_mm:.1f}mm",
                    f"WRIST  d={wd/px_per_mm:6.1f}mm  v={wv/px_per_mm:6.1f}mm/s  a={wa/px_per_mm:7.1f}mm/s2",
                    f"THUMB  d={td/px_per_mm:6.1f}mm  v={tv/px_per_mm:6.1f}mm/s  a={ta/px_per_mm:7.1f}mm/s2",
                    f"INDEX  d={id_/px_per_mm:6.1f}mm  v={iv/px_per_mm:6.1f}mm/s  a={ia/px_per_mm:7.1f}mm/s2",
                ]
                overlay_text(frame, lines)
                break

        if not detected:
            overlay_text(frame, [f"Frame: {frame_idx:5d} – roka ni zaznana"])

        return frame

    # Obdelaj kalibracijske okvirje
    for frame in all_frames:
        processed = process_frame(frame, frame_idx)
        writer.write(processed)
        if show:
            cv2.imshow("9HPT – Kinematika roke", processed)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                cap.release()
                writer.release()
                hands.close()
                cv2.destroyAllWindows()
                return
        frame_idx += 1

    # Obdelaj preostanek videa
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed = process_frame(frame, frame_idx)
        writer.write(processed)

        if show:
            cv2.imshow("9HPT – Kinematika roke", processed)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_idx += 1
        if frame_idx % 100 == 0:
            elapsed = time.time() - t0
            print(f"[INFO] Obdelano {frame_idx}/{total} okvirjev  ({elapsed:.1f}s)")

    # ── Zaključek ─────────────────────────────────────────────────────────────
    cap.release()
    writer.release()
    hands.close()
    if show:
        cv2.destroyAllWindows()

    print(f"[INFO] Anotiran video shranjen: {out_path}")

    # ── Shrani rezultate ──────────────────────────────────────────────────────
    results = {
        "source":        str(input_source),
        "fps":           fps,
        "px_per_mm":     px_per_mm,
        "calib_auto":    detected_px_per_mm is not None,
        "holes_detected":len(hole_centers),
        "total_frames":  frame_idx,
        "grip_events":   grip.grip_events,
        "trackers": {
            "wrist": wrist_t.to_dict(),
            "thumb": thumb_t.to_dict(),
            "index": index_t.to_dict(),
        }
    }

    json_path = os.path.join(output_dir, "kinematics.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[INFO] Kinematični podatki (JSON): {json_path}")

    csv_path = os.path.join(output_dir, "wrist_kinematics.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "time_s", "x_px", "y_px",
                    "path_mm", "velocity_mm_s", "accel_mm_s2", "grip"])
        grip_frames = {e["frame"]: e["type"] for e in grip.grip_events}
        for i, (t, pos, d, v, a) in enumerate(zip(
                wrist_t.timestamps, wrist_t.positions,
                wrist_t.path_length, wrist_t.velocity, wrist_t.accel)):
            w.writerow([
                i, f"{t:.4f}", pos[0], pos[1],
                f"{d/px_per_mm:.3f}",
                f"{v/px_per_mm:.3f}",
                f"{a/px_per_mm:.3f}",
                grip_frames.get(i, ""),
            ])
    print(f"[INFO] Wrist CSV: {csv_path}")

    # Izpiši grip evenimente
    if grip.grip_events:
        print(f"\n── GRIP DOGODKI ({len(grip.grip_events)}) ──────────────────")
        for e in grip.grip_events:
            print(f"  {e['type']:8s}  t={e['time_s']:.2f}s  frame={e['frame']}")

    print("\n── POVZETEK ──────────────────────────────────────────────────")
    for name, tracker in [("Wrist", wrist_t), ("Thumb", thumb_t), ("Index", index_t)]:
        if tracker.path_length:
            max_v   = max(tracker.velocity) / px_per_mm
            avg_v   = np.mean(tracker.velocity) / px_per_mm
            total_d = tracker.path_length[-1] / px_per_mm
            print(f"  {name:6s}: skupna pot={total_d:.1f} mm  "
                  f"avg v={avg_v:.1f} mm/s  max v={max_v:.1f} mm/s")
    print("──────────────────────────────────────────────────────────────\n")

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="9HPT Robotski vid – kinematika roke")
    parser.add_argument("--input",     default="0")
    parser.add_argument("--output",    default="data/results")
    parser.add_argument("--px_per_mm", type=float, default=None,
                        help="Ročni umeritveni faktor (privzeto: avtomatsko)")
    parser.add_argument("--no-show",   action="store_true")
    args = parser.parse_args()

    source = int(args.input) if args.input.isdigit() else args.input

    process_video(
        input_source=source,
        output_dir=args.output,
        px_per_mm=args.px_per_mm,
        show=not args.no_show,
    )
